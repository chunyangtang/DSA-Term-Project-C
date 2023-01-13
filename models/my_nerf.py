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
        # 存储立方体大小
        self.N, _ = pts_xyz.shape
        self.N = round(self.N ** (1/3))
        # load from the disk if exists
        # try:
        #     checkpoint = torch.load("checkpoints/sigma_color_save.pth")
        #     self.volumes_sigma = checkpoint["volumes_sigma"]
        #     self.volumes_color = checkpoint["volumes_color"]
        #     assert self.volumes_sigma.shape == (self.N, self.N, self.N)
        #     print('VALID CHECKPOINT FOUND',
        #           self.volumes_sigma.shape, self.volumes_color.shape)
        #     return
        # except:
        # 存储体积中各点的sigma和color
        self.volumes_sigma = sigma.reshape((self.N, self.N, self.N))
        self.volumes_color = color.reshape((self.N, self.N, self.N, 3))
            # save to the disk
            # checkpoint = {
            #     "volumes_sigma": self.volumes_sigma,
            #     "volumes_color": self.volumes_color
            # }
            # torch.save(checkpoint, "checkpoints/sigma_color_save.pth")
            # print('NEW CHECKPOINT CREATED',
            #       self.volumes_sigma.shape, self.volumes_color.shape)

    def query(self, pts_xyz):
        # 获取带查询坐标的数量
        N_xyz, _ = pts_xyz.shape
        # 初始化sigma和color
        sigma = torch.zeros(N_xyz, 1, device=pts_xyz.device)
        color = torch.zeros(N_xyz, 3, device=pts_xyz.device)
        # 获取待查询坐标的x,y,z分量
        X_index = ((pts_xyz[:, 0] + 0.125) * 4 *
                   self.N).clamp(0, self.N-1).long()
        Y_index = ((pts_xyz[:, 1] - 0.75) * 4 *
                   self.N).clamp(0, self.N-1).long()
        Z_index = ((pts_xyz[:, 2] + 0.125) * 4 *
                   self.N).clamp(0, self.N-1).long()
        # 根据索引的x,y,z坐标，获取对应的sigma和color
        sigma[:, 0] = self.volumes_sigma[X_index, Y_index, Z_index]
        color[:, :] = self.volumes_color[X_index, Y_index, Z_index]

        return sigma, color
