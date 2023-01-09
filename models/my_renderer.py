import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import mcubes
import time

def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    # cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples, device=weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class MyNerfRenderer:
    def __init__(self,
                 my_nerf,
                 n_samples,
                 n_importance,
                 perturb):
        self.my_nerf = my_nerf
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.perturb = perturb

    def render(self, rays_o, rays_d, near, far, background_rgb=None):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=rays_o.device)
        z_vals = near + (far - near) * z_vals[None, :]
    
        n_samples = self.n_samples
        perturb = self.perturb

        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1], device=rays_o.device) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

        # Up sample
        if self.n_importance > 0:
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
            dirs = rays_d[:, None, :].expand(pts.shape)
            pts = pts.reshape(-1, 3)
            dirs = dirs.reshape(-1, 3)

            sigma, sampled_color = self.my_nerf.query(pts)
            sigma = sigma.reshape(batch_size, n_samples)
            sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
            
            dists = z_vals[..., 1:] - z_vals[..., :-1]
            dists = torch.cat([dists, torch.Tensor([1/128]).expand(dists[..., :1].shape).to(rays_o.device)], -1)
            alpha = 1.0 - torch.exp(-F.softplus(sigma.reshape(batch_size, n_samples)) * dists)
            coarse_weights = alpha * torch.cumprod(
                torch.cat([torch.ones([batch_size, 1], device=rays_o.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
            coarse_color = (sampled_color * coarse_weights[:, :, None]).sum(dim=1)
            if background_rgb is not None:
                coarse_color = coarse_color + background_rgb * (1.0 - coarse_weights.sum(dim=-1, keepdim=True))

            with torch.no_grad():
                new_z_vals = sample_pdf(z_vals, coarse_weights, self.n_importance, det=True).detach()
            z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
            z_vals, index = torch.sort(z_vals, dim=-1)

            n_samples = self.n_samples + self.n_importance
            pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
            dirs = rays_d[:, None, :].expand(pts.shape)
            pts = pts.reshape(-1, 3)
            dirs = dirs.reshape(-1, 3)

            sigma, sampled_color = self.my_nerf.query(pts)
            sigma = sigma.reshape(batch_size, n_samples)
            sampled_color = sampled_color.reshape(batch_size, n_samples, 3)

            dists = z_vals[..., 1:] - z_vals[..., :-1]
            dists = torch.cat([dists, torch.Tensor([1/128]).expand(dists[..., :1].shape).to(rays_o.device)], -1)
            alpha = 1.0 - torch.exp(-F.softplus(sigma.reshape(batch_size, n_samples)) * dists)
            fine_weights = alpha * torch.cumprod(
                torch.cat([torch.ones([batch_size, 1], device=rays_o.device), 1. - alpha + 1e-7], -1), -1)[:, :-1]
            fine_color = (sampled_color * fine_weights[:, :, None]).sum(dim=1)
            if background_rgb is not None:
                fine_color = fine_color + background_rgb * (1.0 - fine_weights.sum(dim=-1, keepdim=True))


        return {
            'fine_color': fine_color,
            'coarse_color': coarse_color,
            'fine_weights': fine_weights,
            'coarse_weights': coarse_weights,
            'z_vals': z_vals,
        }
