import os
import time
import logging
import argparse
import numpy as np
import cv2 as cv
import trimesh
import mcubes
import torch
import torch.nn.functional as F
from icecream import ic
from tqdm import tqdm
from pyhocon import ConfigFactory
from models.fields import NeRF
from models.my_dataset import Dataset
from models.my_nerf import MyNeRF, CheatNeRF
from models.my_renderer import MyNerfRenderer
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class Runner:
    def __init__(self, conf_path, mode='render', case='CASE_NAME', is_continue=False):
        self.device = torch.device('cpu')
        # self.device = torch.device('cuda:0')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace(
            'CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        self.checkpoint_dir = self.conf['general.checkpoint_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'], self.device)
        self.iter_step = 0
        self.use_white_bkgd = self.conf.get_bool('train.use_white_bkgd')

        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        self.coarse_nerf = NeRF(
            **self.conf['model.coarse_nerf']).to(self.device)
        self.fine_nerf = NeRF(**self.conf['model.fine_nerf']).to(self.device)
        self.my_nerf = MyNeRF()
        self.renderer = MyNerfRenderer(self.my_nerf,
                                       **self.conf['model.nerf_renderer'])
        self.load_checkpoint(r'./nerf_model.pth', absolute=False)

    def load_checkpoint(self, checkpoint_name, absolute=False):
        if absolute:
            checkpoint = torch.load(checkpoint_name, map_location=self.device)
        else:
            checkpoint = torch.load(os.path.join(
                self.checkpoint_dir, checkpoint_name), map_location=self.device)
        self.coarse_nerf.load_state_dict(checkpoint['coarse_nerf'])
        self.fine_nerf.load_state_dict(checkpoint['fine_nerf'])
        logging.info('End')

    def use_nerf(self):
        self.my_nerf = CheatNeRF(self.fine_nerf)
        self.renderer = MyNerfRenderer(self.my_nerf,
                                       **self.conf['model.nerf_renderer'])

    def save(self, timecompare_mode=False):
        RS = 128
        pts_xyz = torch.zeros((RS, RS, RS, 3))
        if not timecompare_mode:
            for i in tqdm(range(RS)):
                for j in range(RS):
                    pts_xyz[:, i, j, 0] = torch.linspace(-0.125, 0.125, RS)
                    pts_xyz[i, :, j, 1] = torch.linspace(0.75, 1.0, RS)
                    pts_xyz[i, j, :, 2] = torch.linspace(-0.125, 0.125, RS)
            pts_xyz = pts_xyz.reshape((RS*RS*RS, 3))
            batch_size = 1024
            # load from the disk if exists
            try:
                checkpoint = torch.load("checkpoints/sigma_color.pth")
                sigma = checkpoint["sigma"]
                color = checkpoint["color"]
                assert sigma.shape == (RS*RS*RS, 1)
                print('CHECKPOINT LOADED, sigma ',
                      sigma.shape, 'color', color.shape)
            except:
                sigma = torch.zeros((RS*RS*RS, 1))
                color = torch.zeros((RS*RS*RS, 3))
                for batch in tqdm(range(0, pts_xyz.shape[0], batch_size)):
                    batch_pts_xyz = pts_xyz[batch:batch+batch_size]
                    net_sigma, net_color = self.fine_nerf(
                        batch_pts_xyz, torch.zeros_like(batch_pts_xyz))
                    sigma[batch:batch+batch_size] = net_sigma
                    color[batch:batch+batch_size] = net_color
                # save to the disk
                checkpoint = {
                    "sigma": sigma,
                    "color": color
                }
                torch.save(checkpoint, "checkpoints/sigma_color.pth")
                print('NO VALID CHECKPOINT, GENERATED A NEW ONE')
        else:
            for i in range(RS):
                for j in tqdm(range(RS)):
                    pts_xyz[:, i, j, 0] = torch.linspace(-0.125, 0.125, RS)
                    pts_xyz[i, :, j, 1] = torch.linspace(0.75, 1.0, RS)
                    pts_xyz[i, j, :, 2] = torch.linspace(-0.125, 0.125, RS)
            pts_xyz = pts_xyz.reshape((RS*RS*RS, 3))
            batch_size = 1024

            sigma = torch.zeros((RS*RS*RS, 1))
            color = torch.zeros((RS*RS*RS, 3))
            for batch in tqdm(range(0, pts_xyz.shape[0], batch_size)):
                batch_pts_xyz = pts_xyz[batch:batch+batch_size]
                net_sigma, net_color = self.fine_nerf(
                    batch_pts_xyz, torch.zeros_like(batch_pts_xyz))
                sigma[batch:batch+batch_size] = net_sigma
                color[batch:batch+batch_size] = net_color

        self.my_nerf.save(pts_xyz, sigma, color)

    def render_video(self, timecompare_mode=False, filename='show'):
        images = []
        resolution_level = 1
        n_frames = 90
        for idx in tqdm(range(n_frames)):
            rays_o, rays_d = self.dataset.gen_rays_at(
                idx, resolution_level=resolution_level)
            H, W, _ = rays_o.shape
            rays_o = rays_o.reshape(-1, 3).split(1024)
            rays_d = rays_d.reshape(-1, 3).split(1024)

            out_rgb_fine = []

            for rays_o_batch, rays_d_batch in tqdm(zip(rays_o, rays_d)):
                near, far = self.dataset.near_far_from_sphere(
                    rays_o_batch, rays_d_batch)
                background_rgb = torch.ones(
                    [1, 3], device=self.device) if self.use_white_bkgd else None

                render_out = self.renderer.render(rays_o_batch,
                                                    rays_d_batch,
                                                    near,
                                                    far,
                                                    background_rgb=background_rgb)

                def feasible(key): return (key in render_out) and (
                    render_out[key] is not None)

                if feasible('fine_color'):
                    out_rgb_fine.append(
                        render_out['fine_color'].detach().cpu().numpy())

                del render_out

            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape(
                [H, W, 3]) * 256).clip(0, 255)
            img_fine = cv.resize(cv.flip(img_fine, 0), (512, 512))
            images.append(img_fine)
            if not timecompare_mode:
                os.makedirs(os.path.join(
                    self.base_exp_dir, 'render'), exist_ok=True)
                cv.imwrite(os.path.join(self.base_exp_dir,  'render',
                                        '{}.jpg'.format(idx)), img_fine)
            else:
                os.makedirs(os.path.join(
                    self.base_exp_dir, 'render_compare'), exist_ok=True)
                cv.imwrite(os.path.join(self.base_exp_dir,  'render_compare',
                                        '{}_{}.jpg'.format(filename, idx)), img_fine)

        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        h, w, _ = images[0].shape
        if not timecompare_mode:
            writer = cv.VideoWriter(os.path.join(self.base_exp_dir,  'render', 'show.mp4'),
                            fourcc, 30, (w, h))
        else:
            writer = cv.VideoWriter(os.path.join(self.base_exp_dir,  'render_compare', '{}.mp4'.format(filename)),
                            fourcc, 30, (w, h))
        for image in tqdm(images):
            image = image.astype(np.uint8)
            writer.write(image)
        writer.release()

    def render_image(self, idx=0):
        """
        Render an image at a given index in the dataset.
        """
        resolution_level = 1
        rays_o, rays_d = self.dataset.gen_rays_at(
            idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(1024)
        rays_d = rays_d.reshape(-1, 3).split(1024)

        out_rgb_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(
                rays_o_batch, rays_d_batch)
            background_rgb = torch.ones(
                [1, 3], device=self.device) if self.use_white_bkgd else None

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              background_rgb=background_rgb)

            def feasible(key): return (key in render_out) and (
                render_out[key] is not None)

            if feasible('fine_color'):
                out_rgb_fine.append(
                    render_out['fine_color'].detach().cpu().numpy())

            del render_out

        img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape(
            [H, W, 3]) * 256).clip(0, 255)
        img_fine = cv.resize(cv.flip(img_fine, 0), (512, 512))
        os.makedirs(os.path.join(self.base_exp_dir, 'render'), exist_ok=True)
        cv.imwrite(os.path.join(self.base_exp_dir,  'render',
                   'custom_angle', '{}.jpg'.format(idx)), img_fine)

    def mcube_save(self, threshold=0.0):
        """
        Run marching cube algorithm and save it as obj file.
        """
        # Converting sigma to numpy array
        sigma = self.my_nerf.volumes_sigma.detach().cpu().numpy()
        # Marching cubes
        vertices, triangles = mcubes.marching_cubes(sigma, threshold)
        # Save mesh
        os.makedirs(os.path.join(self.base_exp_dir, 'mesh'), exist_ok=True)
        mcubes.export_obj(vertices, triangles, os.path.join(
            self.base_exp_dir, 'mesh', 'mesh.obj'))

    def mcube(self, threshold=0.0):
        """
        Run marching cube algorithm to extract mesh from volume.
        """
        # Converting sigma to numpy array
        sigma = self.my_nerf.volumes_sigma.detach().cpu().numpy()
        # Marching cubes
        vertices, triangles = mcubes.marching_cubes(sigma, threshold)
        # Show mesh
        mesh = trimesh.Trimesh(vertices / 128 - .5, triangles)
        mesh.show()
        # Save mesh
        os.makedirs(os.path.join(self.base_exp_dir, 'mesh'), exist_ok=True)
        mcubes.export_obj(vertices, triangles, os.path.join(
            self.base_exp_dir, 'mesh', 'mesh.obj'))

    def time_compare(self, idx=0, threshold=0.0):
        """
        Compare the time of rendering using different methods.
        """
        # Rendering using NeRF
        start = time.time()
        self.use_nerf()
        self.render_video(timecompare_mode=True, filename='neural')
        end = time.time()
        print('Time of Nerual Rendering: {}s'.format(end - start))
        # Rendering using Voxels
        start = time.time()
        self.save(timecompare_mode=True)
        self.render_video(timecompare_mode=True, filename='voxel')
        end = time.time()
        print('Time of Voxel Rendering: {}s'.format(end - start))


if __name__ == '__main__':

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='render')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--dataroot', type=str, default='')

    args = parser.parse_args()
    runner = Runner(args.conf, args.mode, args.case, args.is_continue)

    if args.mode == 'render':
        runner.save()
        runner.render_video()
    elif args.mode == 'test':
        runner.use_nerf()
        runner.render_video()
    elif args.mode == 'mcube':
        runner.save()
        runner.mcube_save(args.mcube_threshold)
    elif args.mode == 'mcube_show':
        runner.save()
        runner.mcube(args.mcube_threshold)
    elif args.mode == 'time_compare':
        runner.time_compare()
