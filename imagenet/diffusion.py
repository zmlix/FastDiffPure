import torch
import torch.nn as nn
from torchvision.utils import save_image
import os

from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

# device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

class Args:
    image_size=256
    num_channels=256
    num_res_blocks=2
    num_heads=4
    num_heads_upsample=-1
    num_head_channels=64
    attention_resolutions="32,16,8"
    channel_mult=""
    dropout=0.0
    class_cond=False
    use_checkpoint=False
    use_scale_shift_norm=True
    resblock_updown=True
    use_fp16=False
    use_new_attention_order=False
    clip_denoised=True
    num_samples=10000
    batch_size=16
    use_ddim=False
    model_path=""
    classifier_path=""
    classifier_scale=1.0
    learn_sigma=True
    diffusion_steps=1000
    noise_schedule="linear"
    timestep_respacing=None
    use_kl=False
    predict_xstart=False
    rescale_timesteps=False
    rescale_learned_sigmas=False


class DiffusionRobustModel(nn.Module):
    def __init__(self,device):
        super().__init__()
        self.device = device
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            torch.load("256x256_diffusion_uncond.pt", map_location=torch.device(self.device))
        )
        model.eval().to(self.device)

        self.model = model 
        self.diffusion = diffusion

    def denoise(self, x_start, t, x_mae, multistep = False, s=None, **kwags):
        if x_mae is None:
            return self.denoise2(x_start, t, x_mae, multistep=multistep)
        # print(t)
        t_batch = torch.tensor([t] * len(x_start)).to(self.device)
        noise = torch.randn_like(x_start)
        x_t_start = self.diffusion.q_sample(x_start=x_start, t=t_batch, noise=noise)
        x_mae_start = self.diffusion.q_sample(x_start=x_mae, t=t_batch, noise=noise)
        # x_t_start_ = x_t_start.clone().detach().requires_grad_(True)
        x_t_start.requires_grad=True

        # loss = 1 - ssim(x_t_start, x_mae_start, data_range=1, size_average=False)
        mse_loss = nn.MSELoss(reduction='none')
        loss = mse_loss(x_t_start, x_mae_start)

        loss.backward(torch.ones_like(loss))
        grad = x_t_start.grad
        # print(grad.shape)
        # 25000
        S = s * torch.rand_like(x_t_start) * self.diffusion.get_sqrt_one_minus_alphas_cumprod(x_start, t_batch) / self.diffusion.get_sqrt_alphas_cumprod(x_start, t_batch)
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

        sample = (out["mean"] - S*var*grad) + sqrt_var * noise
        # print((s*var*grad).mean(), out["mean"].mean())
        # sample = out["mean"] + sqrt_var * noise

        sample = self.diffusion.p_sample(
                    self.model,
                    sample,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']
        save_image((sample + 1) / 2, os.path.join('./',  'guide_sample.png'))

        return sample

    def denoise2(self, x_start, t, delta = None, multistep=False):
        t_batch = torch.tensor([t] * len(x_start)).to(self.device)

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
                    t_batch = torch.tensor([i] * len(x_start)).to(self.device)
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