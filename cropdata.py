import numpy as np
import os
import sys
import torch
import modules.config as cfg
from modules.data import Preprocessing as pre

dataroot = '/mnt/D/AIProject/mmdetection3d-master/data/kitti'
if len(sys.argv) > 1 and sys.argv[1] != '-':
    dataroot = sys.argv[1]
veloroot = os.path.join(dataroot, 'training/velodyne')
calibroot = os.path.join(dataroot, 'training/calib')
croproot = os.path.join(dataroot, 'training/velodyne_croped')

mode = 'numpy'
if len(sys.argv) > 2 and sys.argv[2] != '-':
    mode = sys.argv[2]

imsize = cfg.imsize[::-1]

if __name__ == '__main__':

    if not os.path.exists(croproot):
        os.mkdir(croproot)

    allSet = ['%06d' % i for i in range(7481)]
    for i, s in enumerate(allSet):
        print(f'\rProcessing: {i + 1}/{7481}', end = '')
        path = os.path.join(veloroot, s + '.bin')
        velo = np.fromfile(path, dtype = 'float32').reshape((-1, 4))
        if mode == 'numpy':
            velo = pre.crop(velo, cfg.velorange)
        elif mode == 'torch':
            velo = torch.Tensor(velo)
            velo = pre.cropTensor(velo, cfg.velorange)
        elif mode == 'torch-cuda':
            velo = torch.Tensor(velo).cuda()
            velo = pre.cropTensor(velo, cfg.velorange)

        path = os.path.join(calibroot, s + '.txt')
        calib = {}
        with open(path, 'r') as f:
            lines = f.read().splitlines()
            l = lines[5].split(' ')
            v2c = np.array(l[1:]).astype('float32').reshape((3, 4))
            v2c = np.concatenate([v2c, [[0, 0, 0, 1]]], axis = 0)
            calib[l[0][:-1]] = v2c
            l = lines[2].split(' ')
            p2 = np.array(l[1:]).astype('float32').reshape((3, 4))
            p2 = np.concatenate([p2, [[0, 0, 0, 1]]], axis = 0)
            calib[l[0][:-1]] = p2
            l = lines[4].split(' ')
            r0 = np.zeros((4, 4))
            r0[:3, :3] = np.array(l[1:]).astype('float32').reshape((3, 3))
            r0[3, 3] = 1
            calib[l[0][:-1]] = r0

        if mode == 'torch':
            for k in calib.keys():
                calib[k] = torch.Tensor(calib[k])
        elif mode == 'torch-cuda':
            for k in calib.keys():
                calib[k] = torch.Tensor(calib[k]).cuda()

        velo = pre.cropToSight(velo, calib, imsize)
        if mode != 'numpy':
            velo = velo.cpu().numpy()
        velo.tofile(os.path.join(croproot, s + '.bin'))