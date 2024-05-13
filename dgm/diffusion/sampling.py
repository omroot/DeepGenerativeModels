

from inspect import isfunction
from einops.layers.torch import Rearrange

import numpy as np

import torch
from torch import nn

import torch.nn.functional as F
from tqdm.auto import tqdm

import matplotlib.pyplot as plt

from dgm.diffusion.utils import exists, default, extract

# forward diffusion (using the nice property)

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


def q_sample(x_start,
             t,
            sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,
             noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )

    return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise


def get_noisy_image(x_start,
                    t,
                    reverse_transform,
                    sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod):
  # add noise
  x_noisy = q_sample(x_start,
                     t,
                     sqrt_alphas_cumprod,
                     sqrt_one_minus_alphas_cumprod,
                     )

  # turn back into PIL image
  noisy_image = reverse_transform(x_noisy.squeeze())

  return noisy_image

def p_losses(denoise_model, x_start, t, sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,noise=None, loss_type="l1"):
    if noise is None:
        noise = torch.randn_like(x_start)

    x_noisy = q_sample(x_start, t, sqrt_alphas_cumprod,
            sqrt_one_minus_alphas_cumprod,noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()

    return loss


@torch.no_grad()
def p_sample(model,
             x,
             t,
             t_index,
             betas,
             sqrt_one_minus_alphas_cumprod,
             sqrt_recip_alphas,
             posterior_variance):
    betas_t = extract(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas, t, x.shape)
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (
            x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise

    # Algorithm 2 (including returning all images)


@torch.no_grad()
def p_sample_loop(model,
                  shape,
                  timesteps,
                  betas,
                  sqrt_one_minus_alphas_cumprod,
                  sqrt_recip_alphas,
                  posterior_variance
                  ):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='sampling loop time step', total=timesteps):
        img = p_sample(model,
                       img,
                       torch.full((b,),
                      i,
                    device=device,
                    dtype=torch.long),
                       i,
                       betas,
                       sqrt_one_minus_alphas_cumprod,
                       sqrt_recip_alphas,
                       posterior_variance
                       )
        imgs.append(img.cpu().numpy())
    return imgs


@torch.no_grad()
def sample(model,
           image_size,
           timesteps,
           batch_size,
           channels,
           betas,
           sqrt_one_minus_alphas_cumprod,
           sqrt_recip_alphas,
           posterior_variance
           ):
    return p_sample_loop(model=model,
                         shape=(batch_size, channels, image_size, image_size),
                         timesteps=timesteps,
                         betas=betas,
                         sqrt_one_minus_alphas_cumprod=sqrt_one_minus_alphas_cumprod,
                         sqrt_recip_alphas=sqrt_recip_alphas,
                         posterior_variance=posterior_variance
                         )