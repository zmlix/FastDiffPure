from enum import Enum
import argparse
import os
import torch
import torchvision
# from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision.utils import save_image
from torch.utils.data import Dataset
from tqdm import tqdm
import foolbox as fb
from autoattack import AutoAttack
from robustbench.utils import load_model
from diffusion import DiffusionRobustModel
from bpda_eot_attack import BPDA_EOT_Attack
import torch.distributed as dist
import apex

os.environ['LOCAL_RANK'] = "0,2,3"
args = None
device = 'cuda'
NROW = 16

class AttackMode(Enum):
    # PrecessorBlind = "PrecessorBlind"
    Plain = "plain"
    AutoAttack = "AutoAttack"
    PGDAttack = "PGDAttack"
    BPDA_EOT = "BPDA_EOT"

class ImagenetDataset(Dataset):
    def __init__(self):
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((256,256), antialias=True)
            # transforms.Normalize(mean, std)
        ])
        
        self.dataset = torchvision.datasets.ImageFolder('/home/mlt01/imagenetval/imagenetval',transform=self.transform)
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # img = self.transform(img)
        return img, label

class DModel(torch.nn.Module):
    def __init__(self, base_model, pure_model, T1, T2, s):
        super().__init__()
        self.T1 = T1
        self.T2 = T2
        self.s = s
        self.base_model = base_model
        self.pure_model = pure_model

    def forward(self, imgs, mode='purify_and_classify'):
        try:
            imgs = imgs.raw
        except:
            pass

        if mode == "purify":
            imgs = Normalize(0.5, 0.5)(imgs)  # [0,1] -> [-1,1]
            p_imgs = diffusion_step(self.pure_model, imgs, self.T1, None)
            p_imgs = diffusion_step(self.pure_model, imgs, self.T2, p_imgs, s=self.s, multistep=False)
            p_imgs = torch.clip((p_imgs + 1) / 2, 0, 1)
            return p_imgs

        if mode == "classify":
            return self.base_model(imgs)

        if mode == "purify_and_classify":
            imgs = Normalize(0.5, 0.5)(imgs)  # [0,1] -> [-1,1]
            # imgs = imgs * 2 - 1
            p_imgs = diffusion_step(self.pure_model, imgs, self.T1, None)
            p_imgs = diffusion_step(self.pure_model, imgs, self.T2, p_imgs, s=self.s, multistep=False)
            p_imgs = torch.clip((p_imgs + 1) / 2, 0, 1)
            return self.base_model(p_imgs)

def get_model(isFoolbox=True):
    # model = load_model(model_name='Standard_R50', dataset='imagenet', threat_model='Linf')
    model = torchvision.models.resnet152(weights='DEFAULT')
    model.eval()  # 验证模型
    # preprocessing = dict(mean=[0.4914, 0.4822, 0.4465],
    #                      std=[0.2023, 0.1994, 0.2010],
    #                      axis=-3)
    bounds = (0, 1)

    if isFoolbox:
        model = fb.PyTorchModel(model, bounds=bounds,device=device) # preprocessing=preprocessing
    return model

def PGDAttack_model(fmodel, imgs, labels, bs=128):
    attack = fb.attacks.LinfPGD(rel_stepsize=0.25, steps=40)
    # attack = fb.attacks.L2PGD()
    save_image(imgs, os.path.join('./', 'raw_cifar10.png'), nrow=NROW)
    # print("true labels", labels)
    # print("pred labels", fmodel(imgs).softmax(-1).argmax(-1))
    # accuracy = fb.utils.accuracy(fmodel, imgs, labels)
    # print("accuracy", accuracy)
    # criterion = fb.criteria.Misclassification(labels)
    adv_images, clipped, _ = attack(fmodel, imgs, labels, epsilons=4 / 255)
    # print("attk labels:", fmodel(clipped).softmax(-1).argmax(-1))
    # accuracy = fb.utils.accuracy(fmodel, clipped, labels)
    # print("adv_accuracy", accuracy)
    save_image(clipped, os.path.join('./', 'adv_cifar10.png'), nrow=NROW)
    return clipped

def autoAttack_model(adversary, imgs, labels, bs=128):
    x_adv = adversary.run_standard_evaluation(imgs, labels, bs=bs)
    save_image(x_adv, os.path.join('./', 'auto_adv_cifar10.png'), nrow=NROW)
    # print(torch.norm(imgs - x_adv, p = 2, dim=(1,2,3)).mean())
    return x_adv

def BPDAEOT_model(adversary, imgs, labels, bs=128):
    class_batch, ims_adv_batch = adversary.attack_all(imgs, labels, batch_size=bs)
    init_acc = float(class_batch[0, :].sum()) / class_batch.shape[1]
    robust_acc = float(class_batch[-1, :].sum()) / class_batch.shape[1]
    print('init acc: {:.2%}, robust acc: {:.2%}'.format(init_acc, robust_acc))
    ims_adv_batch = ims_adv_batch.to(device)
    save_image(ims_adv_batch, os.path.join('./', 'bpda_adv_cifar10.png'), nrow=NROW)
    return ims_adv_batch

def diffusion_step(diffusion_model, imgs, t, delta, multistep=False, s=None):
    # input [-1,1] output [-1,1]
    diffusion_imgs = diffusion_model.denoise(imgs, t, delta, multistep=multistep, s=s)
    # 扩散后的图
    if delta is None:
        save_image((diffusion_imgs + 1) / 2,
                os.path.join('./', 'diff_cifar10.png'),
                nrow=NROW)
    else:
        save_image((diffusion_imgs + 1) / 2,
                os.path.join('./', 'diff_cifar10_guided.png'),
                nrow=NROW)
    return diffusion_imgs

def attack(dataloader, batch_size, attack_mode: AttackMode, T1, T2, s):
    print(f"{attack_mode.value} attack, T1 = {T1}, T2 = {T2}, s = {s}")
    model = get_model(isFoolbox=True)
    # model = apex.amp.initialize(model)
    diffusion_model = DiffusionRobustModel(device=device)
    # diffusion_model = apex.amp.initialize(diffusion_model)
    dmodel = DModel(model, diffusion_model, T1, T2, s)
    dmodel = apex.amp.initialize(dmodel, opt_level="O1")
    torch.nn.parallel.DistributedDataParallel(dmodel, device_ids=[args.local_rank])
    print("model done!")
    if attack_mode == AttackMode.BPDA_EOT:
        adversary = BPDA_EOT_Attack(dmodel, adv_steps=40)
        attack_model = BPDAEOT_model

    if attack_mode == AttackMode.AutoAttack:
        # L2 0.5 0.6839801126451635
        # adversary = AutoAttack(fmodel, norm='L2', eps=0.5, version='standard', device=device)
        # Linf 10/255 0.4139303578369653
        # Linf 8/255 0.49194030975227926
        adversary = AutoAttack(model,
                            norm='Linf',
                            eps=4 / 255,
                            version='standard',
                            device=device)
        attack_model = autoAttack_model

    if attack_mode == AttackMode.PGDAttack:
        adversary = model
        attack_model = PGDAttack_model

    avg_accuracy = 0
    N = 0

    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for idx, (imgs, labels) in enumerate(tqdmDataLoader):

            N = idx + 1
            imgs = torch.clip(imgs, 0, 1)
            imgs = imgs.to(device)
            labels = labels.to(device)
            if attack_mode == AttackMode.Plain:
                adv_imgs = imgs
            else:
                adv_imgs = attack_model(adversary, imgs, labels, batch_size)
            adv_imgs = Normalize(0.5, 0.5)(adv_imgs)  # [0,1] -> [-1,1]

            accuracy = fb.utils.accuracy(dmodel,
                                        torch.clip((adv_imgs + 1) / 2, 0, 1),
                                        labels)

            avg_accuracy += accuracy

            tqdmDataLoader.set_postfix(ordered_dict={
                "epoch": idx,
                "accuracy": accuracy,
                "avg": avg_accuracy/N
            })
            # break

    print(avg_accuracy / N)


def main(load_batch_size, attack_mode: AttackMode, T1, T2, s):

    val_dataset = ImagenetDataset()

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    # dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=load_batch_size, sampler=val_sampler)
    
    dataloader = torch.utils.data.DataLoader(val_dataset,
                                             load_batch_size,
                                             sampler=val_sampler,
                                             shuffle=False,
                                             num_workers=4)

    # with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
    #     for idx, (imgs, labels) in enumerate(tqdmDataLoader):
    #         print(idx)
    #         save_image(imgs, os.path.join('./', 'imagenet.png'), nrow=NROW)
    #         break

    attack(dataloader, load_batch_size, attack_mode, T1, T2, s)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="pgd")
    parser.add_argument('--T1', type=int, default=115)
    parser.add_argument('--T2', type=int, default=115)
    parser.add_argument('--scale', type=float, default=0.009)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--device', type=str, default='1')
    parser.add_argument('--local-rank', default=-1, type=int,
                    help='node rank for distributed training')
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    attackMode = AttackMode.Plain

    if args.mode == 'pgd':
        attackMode = AttackMode.PGDAttack
    if args.mode == 'bpda':
        attackMode = AttackMode.BPDA_EOT
    if args.mode == 'auto':
        attackMode = AttackMode.AutoAttack

    # device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'


    main(args.bs ,attackMode, args.T1, args.T2, args.scale)
# 110 100 0.008
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 run.py
# 115 115 0.009
# OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 run.py