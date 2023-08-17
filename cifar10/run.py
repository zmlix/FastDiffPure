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
from torchattacks import EOTPGD
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
    PGD_EOT = "PGD_EOT"
    AutoAttack_EOT = "AutoAttack_EOT"

class BModel(torch.nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.base_model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')
    
    def forward(self, imgs, mode='purify_and_classify'):

        if mode == "purify":
            return imgs

        if mode == "classify":
            return self.base_model(imgs)

        if mode == "purify_and_classify":
            return self.base_model(imgs)

class DModel(torch.nn.Module):

    def __init__(self, T, s):
        super().__init__()
        self.T = T
        self.s = s
        self.base_model = BModel()
        self.pure_model = DiffusionPurificationModel(device=device)

    def diffusion_step(self, imgs, t, s=None):
        # input [-1,1] output [-1,1]
        diffusion_imgs = self.pure_model.denoise(imgs, t, s)
        
        if s is None:
            save_image((diffusion_imgs + 1) / 2,
                    os.path.join('./', 'diff_cifar10.png'),
                    nrow=NROW)
        else:
            save_image((diffusion_imgs + 1) / 2,
                    os.path.join('./', 'diff_cifar10_guided.png'),
                    nrow=NROW)
        return diffusion_imgs

    def purify(self, imgs):
        imgs_ = Normalize(0.5, 0.5)(imgs)  # [0,1] -> [-1,1]
        p_imgs = self.diffusion_step(imgs_, self.T, self.s)
        p_imgs = torch.clip((p_imgs + 1) / 2, 0, 1) # [-1,1] -> [0,1]
        del imgs_
        return p_imgs

    def forward(self, imgs, mode='purify_and_classify'):

        if mode == "purify":
            return self.purify(imgs)

        if mode == "classify":
            return self.base_model(imgs)

        if mode == "purify_and_classify":
            p_imgs = self.purify(imgs)
            return self.base_model(p_imgs)

def get_model(isFoolbox=True):
    model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')
    # model.eval()
    bounds = (0, 1)

    if isFoolbox:
        model = fb.PyTorchModel(model, bounds=bounds,device=device)
    return model

def PGDAttack_model(fmodel, imgs, labels, bs=128):
    attack = fb.attacks.LinfPGD(rel_stepsize=0.25, steps=10)
    # attack = fb.attacks.L2PGD()
    # save_image(imgs, os.path.join('./', 'raw_cifar10.png'), nrow=NROW)
    # print("true labels", labels)
    # print("pred labels", fmodel(imgs).softmax(-1).argmax(-1))
    # accuracy = fb.utils.accuracy(fmodel, imgs, labels)
    # print("accuracy", accuracy)
    criterion = fb.criteria.Misclassification(labels)
    adv_images, clipped, _ = attack(fmodel, imgs, labels, epsilons=8 / 255)
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
    class_batch, ims_adv_batch = adversary.attack_all(imgs, labels, batch_size=1)
    # init_acc = float(class_batch[0, :].sum()) / class_batch.shape[1]
    # robust_acc = float(class_batch[-1, :].sum()) / class_batch.shape[1]
    # print('init acc: {:.2%}, robust acc: {:.2%}'.format(init_acc, robust_acc))
    ims_adv_batch = ims_adv_batch.to(device)
    save_image(ims_adv_batch, os.path.join('./', 'bpda_adv_cifar10.png'), nrow=NROW)
    return ims_adv_batch

def PGDEOT_model(model, imgs, labels, bs=128):
    attack = EOTPGD(model, eps=8/255, alpha=args.alpha, steps=20, eot_iter=20 , random_start=True)
    adv_images = attack(imgs, labels)
    save_image(adv_images, os.path.join('./', 'pgdeot_adv_cifar10.png'), nrow=NROW)
    return adv_images

def AutoAttackEOT_model(adversary, imgs, labels, bs=128):
    x_adv = adversary.run_standard_evaluation(imgs, labels, bs=bs)
    save_image(x_adv, os.path.join('./', 'auto_eot_adv_cifar10.png'), nrow=NROW)
    return x_adv

def attack(dataloader, batch_size, attack_mode: AttackMode, T, s):
    print(f"{attack_mode.value} attack, T = {T}, s = {s}")
    # model = get_model()
    model = BModel().to(device=device).eval()
    # model = fb.PyTorchModel(BModel(), bounds=(0,1),device=device)
    # model = get_model(isFoolbox=False)
    dmodel = DModel(T, s).to(device=device).eval()
    # dmodel = fb.PyTorchModel(DModel(T, s), bounds=(0,1),device=device)
    # dmodel = apex.amp.initialize(dmodel, opt_level="O1")
    # torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
    # torch.nn.parallel.DistributedDataParallel(dmodel, device_ids=[args.local_rank])
    print("model done!")
    if attack_mode == AttackMode.BPDA_EOT:
        adversary = BPDA_EOT_Attack(dmodel, eot_defense_reps=2, adv_steps=40, eot_attack_reps=1)
        attack_model = BPDAEOT_model

    if attack_mode == AttackMode.AutoAttack:
        adversary = AutoAttack(model,
                            norm='Linf',
                            eps=8 / 255,
                            version='standard',
                            verbose=False,
                            device=device)
        attack_model = autoAttack_model

    if attack_mode == AttackMode.PGDAttack:
        adversary = fb.PyTorchModel(model, bounds=(0,1),device=device)
        attack_model = PGDAttack_model

    if attack_mode == AttackMode.PGD_EOT:
        adversary = dmodel
        attack_model = PGDEOT_model

    if attack_mode == AttackMode.AutoAttack_EOT:
        adversary = AutoAttack(dmodel,
                            norm='Linf',
                            eps=8 / 255,
                            verbose=False,
                            version='rand',
                            device=device)
        attack_model = AutoAttackEOT_model

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

            eval_model = fb.PyTorchModel(dmodel, bounds=(0,1),device=device)
            torch.cuda.empty_cache()
            with torch.no_grad():
                accuracy = fb.utils.accuracy(eval_model,
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
    return f"{attack_mode.value} attack, T = {T}, s = {s} | {avg_accuracy / N}"


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

    return attack(dataloader, load_batch_size, attack_mode, T, s)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="plain")
    parser.add_argument('--T', type=int, default=400)
    parser.add_argument('--scale', type=float, default=50000)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--alpha', type=float, default=2/255)
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
    if args.mode == 'pe':
        attackMode = AttackMode.PGD_EOT
    if args.mode == 'ae':
        attackMode = AttackMode.AutoAttack_EOT

    main(args.bs ,attackMode, args.T, args.scale)

# python3 -m torch.distributed.launch --nproc_per_node=4 run.py