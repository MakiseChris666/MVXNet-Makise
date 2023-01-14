from typing import List, Tuple, Dict
import os
import sys
import modules.data.Preprocessing as pre
import torch
import numpy as np
import pandas as pd
from modules import Calc, Config as cfg
import cv2 as cv

dataroot = '../mmdetection3d-master/data/kitti'
if len(sys.argv) > 1 and sys.argv[1] != '#':
    dataroot = sys.argv[1]
veloroot = os.path.join(dataroot, 'training/velodyne_croped')
labelroot = os.path.join(dataroot, 'training/label_2')
calibroot = os.path.join(dataroot, 'training/calib')
imroot = os.path.join(dataroot, 'training/image_2')

rangeMin = torch.Tensor(cfg.velorange[:3])
rangeMax = torch.Tensor(cfg.velorange[3:])
imsize = cfg.imsize[::-1]

def createDataset(splitSet: List[str], needCrop = False) -> \
        Tuple[List[Tuple[np.ndarray, np.ndarray]], List[Tuple[torch.Tensor, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
    """
    Read KITTI data from root.
    @param splitSet: Names of files to read. e.g. ['000000', '000001', ...]
    @param needCrop: Whether to crop point clouds, set to True if directly read from the raw data; Default False
    @return: (x, y, calibs), x is a list of (point cloud, image); y is a list of (bbox, bev of bbox)
            ((None, None) if there's no gtbox), bbox in LiDAR coordinates; calibs is a list of dict containing the
            calibration matrices
    """
    x, y = [], []
    calibs = []
    sum = len(splitSet)
    for i, s in enumerate(splitSet):
        print(f'\rProcessing: {i + 1}/{sum}', end = '')
        path = os.path.join(veloroot, s + '.bin')
        velo = np.fromfile(path, dtype = 'float32').reshape((-1, 4))
        if needCrop:
            velo = pre.crop(velo, cfg.velorange)

        path = os.path.join(imroot, s + '.png')
        img = cv.imread(path)
        img = img[:imsize[1], :imsize[0]]

        path = os.path.join(labelroot, s + '.txt')
        labels = pd.read_csv(path, sep = ' ', index_col = 0, usecols = [0, *[_ for _ in range(8, 15)]])
        labels = labels[labels.index == 'Car'].to_numpy()

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

        if needCrop:
            velo = pre.cropToSight(velo, calib, imsize)
        x.append((velo, img))

        for k in calib.keys():
            calib[k] = torch.Tensor(calib[k])
        calibs.append(calib)

        if len(labels) == 0:
            y.append((None, None))
            continue

        c2v = torch.linalg.inv(calib['Tr_velo_to_cam'])
        labels = torch.Tensor(labels)
        Calc.bboxCam2Lidar(labels, c2v, True)
        inRange = torch.all(labels[:, :3].__lt__(rangeMax[None, ...]), dim = 1) & \
                  torch.all(labels[:, :3].__ge__(rangeMin[None, ...]), dim = 1)
        labels = labels[inRange].contiguous()
        bevs = Calc.bbox3d2bev(labels)
        y.append((labels, bevs))

    print()
    return x, y, calibs
