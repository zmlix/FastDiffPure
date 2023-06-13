from enum import Enum
import argparse
import os
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision.utils import save_image
from tqdm import tqdm
import foolbox as fb
from autoattack import AutoAttack
from robustbench.utils import load_model
from diffusion import DiffusionPurificationModel
from bpda_eot_attack import BPDA_EOT_Attack
import torch.distributed as dist
# import apex

os.environ['LOCAL_RANK'] = "0,1,2,3"
os.environ['OMP_NUM_THREADS'] = "8"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

args = None
device = 'cuda'
NROW = 16

class AttackMode(Enum):
    Plain = "plain"
    AutoAttack = "AutoAttack"
    PGDAttack = "PGDAttack"
    BPDA_EOT = "BPDA_EOT"


class DModel(torch.nn.Module):
    def __init__(self, base_model, pure_model, T, s):
        super().__init__()
        self.T = T
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
            p_imgs = diffusion_step(self.pure_model, imgs, self.T, self.s)
            p_imgs = torch.clip((p_imgs + 1) / 2, 0, 1)
            return p_imgs

        if mode == "classify":
            return self.base_model(imgs)

        if mode == "purify_and_classify":
            imgs = Normalize(0.5, 0.5)(imgs)  # [0,1] -> [-1,1]
            p_imgs = diffusion_step(self.pure_model, imgs, self.T, self.s)
            p_imgs = torch.clip((p_imgs + 1) / 2, 0, 1)
            return self.base_model(p_imgs)

def get_model(isFoolbox=True):
    model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')
    model.eval()
    bounds = (0, 1)

    if isFoolbox:
        model = fb.PyTorchModel(model, bounds=bounds,device=device)
    return model

def PGDAttack_model(fmodel, imgs, labels, bs=128):
    attack = fb.attacks.LinfPGD(rel_stepsize=0.25, steps=10)
    # attack = fb.attacks.L2PGD()
    save_image(imgs, os.path.join('./', 'raw_cifar10.png'), nrow=NROW)
    # print("true labels", labels)
    # print("pred labels", fmodel(imgs).softmax(-1).argmax(-1))
    accuracy = fb.utils.accuracy(fmodel, imgs, labels)
    # print("accuracy", accuracy)
    criterion = fb.criteria.Misclassification(labels)
    adv_images, clipped, _ = attack(fmodel, imgs, labels, epsilons=8 / 255)
    # print("attk labels:", fmodel(clipped).softmax(-1).argmax(-1))
    accuracy = fb.utils.accuracy(fmodel, clipped, labels)
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

def diffusion_step(diffusion_model, imgs, t, s=None):
    # input [-1,1] output [-1,1]
    diffusion_imgs = diffusion_model.denoise(imgs, t, s)
    
    if s is None:
        save_image((diffusion_imgs + 1) / 2,
                os.path.join('./', 'diff_cifar10.png'),
                nrow=NROW)
    else:
        save_image((diffusion_imgs + 1) / 2,
                os.path.join('./', 'diff_cifar10_guided.png'),
                nrow=NROW)
    return diffusion_imgs

def attack(dataloader, batch_size, attack_mode: AttackMode, T, s):
    print(f"{attack_mode.value} attack, T = {T}, s = {s}")
    model = get_model(isFoolbox=True)
    diffusion_model = DiffusionPurificationModel(device=device)
    dmodel = DModel(model, diffusion_model, T, s)
    # dmodel = apex.amp.initialize(dmodel, opt_level="O1")
    torch.nn.parallel.DistributedDataParallel(dmodel, device_ids=[args.local_rank])
    print("model done!")
    if attack_mode == AttackMode.BPDA_EOT:
        adversary = BPDA_EOT_Attack(dmodel, adv_steps=100)
        attack_model = BPDAEOT_model

    if attack_mode == AttackMode.AutoAttack:
        adversary = AutoAttack(model,
                            norm='Linf',
                            eps=8 / 255,
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
            save_image(imgs, os.path.join('./', 'clean_imgs.png'), nrow=NROW)
            if attack_mode == AttackMode.Plain:
                adv_imgs = imgs
            else:
                adv_imgs = attack_model(adversary, imgs, labels, batch_size)

            accuracy = fb.utils.accuracy(dmodel,
                                        torch.clip(adv_imgs, 0, 1),
                                        labels)

            avg_accuracy += accuracy

            tqdmDataLoader.set_postfix(ordered_dict={
                "epoch": idx,
                "accuracy": accuracy,
                "local_rank": args.local_rank,
                "avg": avg_accuracy/N
            })

    print(avg_accuracy / N)


def main(load_batch_size, attack_mode: AttackMode, T, s):

    val_dataset = torchvision.datasets.CIFAR10('CIFAR10',
                                               train=False,
                                               download=True,
                                               transform=Compose([ToTensor()]))
                                               
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    dataloader = torch.utils.data.DataLoader(val_dataset,
                                             load_batch_size,
                                             sampler=val_sampler,
                                             shuffle=False,
                                             num_workers=4)

    attack(dataloader, load_batch_size, attack_mode, T, s)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="plain")
    parser.add_argument('--T', type=int, default=350)
    parser.add_argument('--scale', type=float, default=100000)
    parser.add_argument('--bs', type=int, default=64)
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
    
    main(args.bs ,attackMode, args.T, args.scale)

# python3 -m torch.distributed.launch --nproc_per_node=4 run.py