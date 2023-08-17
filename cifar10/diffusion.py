import torch
import torch.nn as nn
from torchvision.utils import save_image
import os
import time
import math

from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

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

class TemporaryGrad:
    def __enter__(self):
        self.prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        torch.set_grad_enabled(self.prev)


class DiffusionPurificationModel(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        model.load_state_dict(
            # torch.load("cifar10_uncond_50M_500K.pt", map_location=torch.device(self.device))
            torch.load("cifar10_uncond_50M_500K.pt")
        )

        self.model = model
        self.diffusion = diffusion
        self.mse_loss = nn.MSELoss(reduction='none')

    def guide(self, x_t, x_0_t):
        _x_t = x_t.detach().clone()
        _x_t.requires_grad=True
        _x_0_t = x_0_t.detach().clone()
        _x_0_t.requires_grad=True

        loss = self.mse_loss(_x_t, _x_0_t)
        loss.requires_grad_(True)

        loss.backward(torch.ones_like(loss))
        grad = _x_t.grad
        assert grad is not None
        return grad

    def denoise(self, x, t, s=None, **kwgs):
        start_time = time.time()
        t_batch = torch.tensor([t] * len(x)).to(x.device)
        x_t_ = self.diffusion.q_sample(x_start=x, t=t_batch)
        x_pre = self.diffusion.p_sample(
                    self.model,
                    x_t_,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']

        noise = torch.randn_like(x)
        x_t = self.diffusion.q_sample(x_start=x, t=t_batch, noise=noise)
        x_0_t = self.diffusion.q_sample(x_start=x_pre, t=t_batch, noise=noise)

        with TemporaryGrad():
            grad = self.guide(x_t, x_0_t)
            # print(grad.shape)

        s = s or 0
        S = s * self.diffusion.get_sqrt_one_minus_alphas_cumprod(x, t_batch) / self.diffusion.get_sqrt_alphas_cumprod(x, t_batch)
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

    # multi-step
    def denoiseN(self, x, t, s=None, steps = 1, p = 0.8, **kwgs):
        t_batch = torch.tensor([t] * len(x)).to(x.device)
        x_t_ = self.diffusion.q_sample(x_start=x, t=t_batch)
        x_pre = self.diffusion.p_sample(
                    self.model,
                    x_t_,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']

        noise = torch.randn_like(x)
        x_t = self.diffusion.q_sample(x_start=x, t=t_batch, noise=noise)
        x_0_t = self.diffusion.q_sample(x_start=x_pre, t=t_batch, noise=noise)

        def exponential_decay(start_value, decay_rate, num_steps):
            current_value = start_value
            for step in range(num_steps):
                yield current_value
                current_value *= math.exp(-decay_rate)

        for s_ in exponential_decay(s, p, steps):
            with TemporaryGrad():
                grad = self.guide(x_t, x_0_t)
                # print(grad.shape)
            
            S = s_ * self.diffusion.get_sqrt_one_minus_alphas_cumprod(x, t_batch) / self.diffusion.get_sqrt_alphas_cumprod(x, t_batch)
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

            # x_{t-1}
            x_t = (out["mean"] - S*var*grad) + sqrt_var * noise
            t_batch = t_batch - 1
        
        t_batch = t_batch + 1
        sample = self.diffusion.p_sample(
                    self.model,
                    x_t,
                    t_batch,
                    clip_denoised=True
                )['pred_xstart']
        save_image((sample + 1) / 2, os.path.join('./',  'guide_sample.png'))
        return sample
