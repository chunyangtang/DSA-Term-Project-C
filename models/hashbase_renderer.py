import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.my_renderer import sample_pdf

BOX_OFFSETS = torch.tensor([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]])

class HashbaseRenderer:
    def __init__(self,
                 my_nerf,
                 n_samples,
                 n_importance,
                 perturb):
        self.my_nerf = my_nerf
        self.n_samples = n_samples
        self.n_importance = n_importance
        self.perturb = perturb
        self.hashtable = HashTable(log2_hashmap_size=19)

    def hashtable_insert(self, rays_o, rays_d, near, far, background_rgb=None):
        render_output = self.render(rays_o, rays_d, near, far, background_rgb)
        self.hashtable[torch.cat([rays_o, rays_d], dim=1)] = render_output["fine_color"]

    def render(self, rays_o, rays_d, near, far, background_rgb=None):
        # use hashtable first
        fine_color = self.hashtable[torch.cat([rays_o, rays_d], dim=1)]
        # Use voxel query to fill the missing pixels of fine_color
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
            for i in range(batch_size):
                if fine_color[i, 0] == -1:    
                    fine_color[i] = (sampled_color * fine_weights[:, :, None]).sum(dim=1)[i]
            
            if background_rgb is not None:
                fine_color = fine_color + background_rgb * (1.0 - fine_weights.sum(dim=-1, keepdim=True))


        return {
            'fine_color': fine_color,
            'coarse_color': coarse_color,
            'fine_weights': fine_weights,
            'coarse_weights': coarse_weights,
            'z_vals': z_vals,
        }


class HashTable():
    def __init__(self, log2_hashmap_size=19):
        self.log2_hashmap_size = log2_hashmap_size
        self.buckets = -torch.ones([2**log2_hashmap_size, 3])

    def hash(coords, log2_hashmap_size):
        '''
        coords: this function can process upto 7 dim coordinates
        log2T:  logarithm of T w.r.t 2
        '''
        primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

        xor_result = torch.zeros_like(coords)[..., 0]
        for i in range(coords.shape[-1]):
            xor_result ^= coords[..., i]*primes[i]

        return torch.tensor((1<<log2_hashmap_size)-1).to(xor_result.device) & xor_result

    def __getitem__(self, search_key):
        index = hash(search_key, self.log2_hashmap_size)
        return self.buckets[index]

    def __setitem__(self, key, value):
        index = hash(key, self.log2_hashmap_size)
        self.buckets[index] = value
