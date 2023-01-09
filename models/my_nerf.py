import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time

class CheatNeRF():
    def __init__(self, nerf):
        super(CheatNeRF, self).__init__()
        self.nerf = nerf
    
    def query(self, pts_xyz):
        return self.nerf(pts_xyz, torch.zeros_like(pts_xyz))

class MyNeRF():
    def __init__(self):
        super(MyNeRF, self).__init__()

    def save(self, pts_xyz, sigma, color):
        pass
    
    def query(self, pts_xyz):
        N, _ = pts_xyz.shape
        sigma = torch.zeros(N, 1, device=pts_xyz.device)
        color = torch.zeros(N, 3, device=pts_xyz.device)

        return sigma, color


