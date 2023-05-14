import torch
import torch.nn as nn
import torchvision
import foolbox as fb
from torchvision.utils import save_image
import os
from torchvision.transforms import ToTensor, Compose


from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

# from pytorch_msssim import ssim, ms_ssim

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

class Args:
    image_size=32
    num_channels=128
    num_res_blocks=3
    num_heads=4
    num_heads_upsample=-1
    attention_resolutions="16,8"
    dropout=0.3
    learn_sigma=True
    sigma_small=False
    class_cond=False
    diffusion_steps=4000
    noise_schedule="cosine"
    timestep_respacing=""
    use_kl=False
    predict_xstart=False
    rescale_timesteps=True
    rescale_learned_sigmas=True
    use_checkpoint=False
    use_scale_shift_norm=True


class DiffusionRobustModel(nn.Module):
    def __init__(self):
        super().__init__()
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            torch.load("cifar10_uncond_50M_500K.pt", map_location=torch.device(device))
        )
        model.eval().to(device)

        self.model = model
        self.diffusion = diffusion

    def denoise(self, x_start, t, x_guid, multistep = False, **kwags):
        if x_guid is None:
            return self.denoise2(x_start, t, x_guid, multistep=multistep)
        # print(t)
        t_batch = torch.tensor([t] * len(x_start)).to(device)
        noise = torch.randn_like(x_start)
        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)
        x_mae_start = self.diffusion.q_sample(x_start=x_guid, t=t_batch, noise=noise)
        x_t_start_ = x_t_start.clone().detach().requires_grad_(True)
        # x_t_start_.requires_grad=True

        # loss = 1 - ssim(x_t_start, x_mae_start, data_range=1, size_average=False)
        mse_loss = nn.MSELoss(reduction='none')
        loss = mse_loss(x_t_start_, x_mae_start)

        loss.backward(torch.ones_like(loss))
        grad = x_t_start_.grad
        # print(grad.shape)
        # S = 50000
        S = 0.01 * torch.rand_like(x_t_start)
        s = S * self.diffusion.get_sqrt_one_minus_alphas_cumprod(x_start, t_batch) / self.diffusion.get_sqrt_alphas_cumprod(x_start, t_batch)
        # print("s", s.max(), s.min())
        out = self.diffusion.p_mean_variance(
            self.model,
            x_t_start,
            t_batch,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None
        )
        var = torch.exp(out["log_variance"])
        sqrt_var = torch.exp(0.5 * out["log_variance"])
        # noise = torch.randn_like(x_start)

        # sample = (out["mean"] - s*var*grad) + sqrt_var * noise
        sample = out["mean"] + sqrt_var * noise

        sample = self.diffusion.p_sample(
                    self.model,
                    sample,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']
        save_image((sample + 1) / 2, os.path.join('./',  'guide_sample.png'))

        return sample

    def denoise2(self, x_start, t, delta = None, multistep=False):
        t_batch = torch.tensor([t] * len(x_start)).to(device)

        noise = torch.randn_like(x_start)

        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)
        # save_image(x_t_start, os.path.join('./',  'diffusion_noise_cifar10.png'), nrow=NROW)
        if delta is not None:
            delta = self.diffusion.q_sample(x_start=delta, t=t_batch//2, noise=noise)
            x_t_start = x_t_start - delta

        with torch.no_grad():
            if multistep:
                out = x_t_start
                for i in range(t)[::-1]:
                    print(i)
                    t_batch = torch.tensor([i] * len(x_start)).to(device)
                    out = self.diffusion.p_sample(
                        self.model,
                        out,
                        t_batch,
                        clip_denoised=True
                    )['sample']
            else:
                out = self.diffusion.p_sample(
                    self.model,
                    x_t_start,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']

        return out


if __name__ == "__main__":
    N = 4
    NROW = 2
    load_batch_size = 128
    model = DiffusionRobustModel().to(device)
    val_dataset = torchvision.datasets.CIFAR10('CIFAR10', train=False, download=True, transform=Compose([ToTensor()]))
    dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=True, num_workers=4)

    imgs = torch.stack([val_dataset[i][0] for i in range(N)]).to(device)
    labels = torch.tensor([val_dataset[i][1] for i in range(N)]).to(device)

    save_image(imgs, os.path.join('./',  'raw_cifar10.png'), nrow=NROW)

    x_in = imgs * 2 -1
    imgs = model.denoise(x_in, 0, multistep=True)
    save_image((imgs + 1) / 2, os.path.join('./',  'diffusion_cifar10.png'), nrow=NROW)


    # diffusion_imgs = model.denoise_imgs(imgs, 0)

