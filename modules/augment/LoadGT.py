import pickle as pkl
import os
import numpy as np
import cv2 as cv
from modules.data.Load import readCalib
from modules.config import args

dataroot = '../mmdetection3d-master/data/kitti'
if len(args) > 0:
    dataroot = args[0]
calibroot = os.path.join(dataroot, 'training/calib')

def readGTInfo():
    root = os.path.join(dataroot, 'training/gtdatabase/gtinfo.pkl')
    with open(root, 'rb') as f:
        info = pkl.load(f)
    return info

def getGTByInfo(info, cls):
    gtroot = os.path.join(dataroot, 'training/gtdatabase/' + cls)
    veloPath = os.path.join(gtroot, info['velo'])
    imgPath = os.path.join(gtroot, info['image'])
    maskPath = os.path.join(gtroot, info['mask'])
    velo = np.fromfile(veloPath, dtype = 'float32').reshape((-1, 4))
    img = cv.imread(imgPath)
    mask = np.load(maskPath)
    maskbbox = info['maskbbox']
    bbox2d = info['bbox2d']
    bbox3d = info['bbox3d']
    calibPath = os.path.join(calibroot, info['id'] + '.txt')
    calib = readCalib(calibPath)
    return velo, img, mask, maskbbox, bbox2d, bbox3d, calib

def getAllGT(targetCls):
    gtinfo = readGTInfo()
    res = {}
    print('Loading gt database...')
    for c in targetCls:
        cur = []
        print('Loading gt class', c)
        length = len(gtinfo[c])
        for i, info in enumerate(gtinfo[c]):
            print(f'\rLoading {i + 1}/{length}', end = ' ')
            velo, img, mask, maskbbox, bbox2d, bbox3d, calib = getGTByInfo(info, c)
            cur.append({
                'velo': velo,
                'image': img,
                'mask': mask,
                'maskbbox': maskbbox,
                'bbox2d': bbox2d,
                'bbox3d': bbox3d,
                'calib': calib
            })
        res[c] = cur
        print()
    return res