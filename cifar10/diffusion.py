import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
import time

from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

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


class DiffusionPurificationModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            torch.load("cifar10_uncond_50M_500K.pt", map_location=torch.device(self.device))
        )
        model.eval().to(self.device)

        self.model = model
        self.diffusion = diffusion

    def denoise(self, x, t, s=None, **kwgs):
        start_time = time.time()
        t_batch = torch.tensor([t] * len(x)).to(self.device)
        x_t_ = self.diffusion.q_sample(x_start=x, t=t_batch)
        x_0 = self.diffusion.p_sample(
                    self.model,
                    x_t_,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']

        noise = torch.randn_like(x)
        x_t = self.diffusion.q_sample(x_start=x, t=t_batch, noise=noise)
        x_0_t = self.diffusion.q_sample(x_start=x_0, t=t_batch, noise=noise)
        x_t.requires_grad=True

        mse_loss = nn.MSELoss(reduction='none')
        loss = mse_loss(x_t, x_0_t)

        loss.backward(torch.ones_like(loss))
        grad = x_t.grad
        # print(grad.shape)
        # 100000
        s = s or 0
        S = s * torch.rand_like(x_t) * self.diffusion.get_sqrt_one_minus_alphas_cumprod(x, t_batch) / self.diffusion.get_sqrt_alphas_cumprod(x, t_batch)
        # print("s", s.max(), s.min())
        out = self.diffusion.p_mean_variance(
            self.model,
            x_t,
            t_batch,
            clip_denoised=True,
            denoised_fn=None,
            model_kwargs=None
        )
        var = torch.exp(out["log_variance"])
        sqrt_var = torch.exp(0.5 * out["log_variance"])
        noise = torch.randn_like(x)

        sample = (out["mean"] - S*var*grad) + sqrt_var * noise
        # print((s*var*grad).mean(), out["mean"].mean())
        # sample = out["mean"] + sqrt_var * noise

        sample = self.diffusion.p_sample(
                    self.model,
                    sample,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']
        end_time = time.time()
        # print(f"time: {end_time - start_time}")
        save_image((sample + 1) / 2, os.path.join('./',  'guide_sample.png'))
        return sample
