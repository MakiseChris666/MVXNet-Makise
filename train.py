import random
from modules import Config as cfg
import sys
import torch
import os
import time
from modules.Calc import bbox3d2bev, classifyAnchors
from modules.data import Load as load, Preprocessing as pre
from modules.augment.Augment import augmentTargetClasses
from modules.augment.LoadGT import getAllGT
from modules.utils import lidar2Img
from MVXNet import MVXNet
from modules.voxelnet import VoxelLoss
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

device = cfg.device
dataroot = '../mmdetection3d-master/data/kitti'
if len(sys.argv) > 1 and sys.argv[1] != '-':
    dataroot = sys.argv[1]
trainInfoPath = os.path.join(dataroot, 'ImageSets/train.txt')
testInfoPath = os.path.join(dataroot, 'ImageSets/val.txt')

def cputask(data, anchorBevs, gtwithinfo):
    pcd, img, bbox2d, bbox3d, bev, calib = data
    augpcd, augcalib, img, bbox3d, bev = augmentTargetClasses(pcd, img, bbox2d, bbox3d, bev, gtwithinfo, ['Car'], [12])
    bbox3d = bbox3d['Car']
    bev = bev['Car']
    pcd = torch.Tensor(pcd)
    proj = lidar2Img(pcd, calib, True)
    proj = proj[:, [1, 0]]
    pcd = torch.concat([pcd, proj], dim = 1)
    pcd = pcd.numpy()
    pcdxy = [pcd]
    for ap, ac in zip(augpcd, augcalib):
        proj = lidar2Img(ap, ac, True)
        proj = proj[:, ::-1]
        ap = np.concatenate([ap, proj], axis = 1)
        pcdxy.append(ap)
    pcd = np.concatenate(pcdxy, axis = 0)

    voxel, idx = pre.group(pcd, cfg.velorange, cfg.voxelsize, cfg.samplenum)
    if bev is not None:
        pi, ni, gi = classifyAnchors(bev, bbox3d[:, [0, 1]], anchorBevs, cfg.velorange, 0.45, 0.6)
    else:
        pi, ni, gi, l = None, None, None, None
    return voxel, idx, img, bbox3d, bev, pi, ni, gi, calib

def train(processPool):

    with open(trainInfoPath, 'r') as f:
        trainSet = f.read().splitlines()

    trainDataSet = load.createDataset(trainSet)
    gtwithinfo = getAllGT(['Car'])

    anchors = pre.createAnchors(cfg.voxelshape[0] // 2, cfg.voxelshape[1] // 2,
                                          cfg.velorange, cfg.carsize)
    anchorBevs = bbox3d2bev(anchors.reshape(anchors.shape[:2] + (-1, 7)))
    model = MVXNet()
    criterion = VoxelLoss()
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001, eps = cfg.eps)
    # torch.autograd.set_detect_anomaly(True)

    model = model.to(device)
    anchors = anchors.to(device)
    criterion = criterion.to(device)
    imsize = torch.Tensor(cfg.imsize).to(device)
    if cfg.half:
        anchors = anchors.half()

    forwardTime = 0
    lossTime = 0
    backwardTime = 0
    allTime = 0

    if len(sys.argv) > 2:
        iterations = int(sys.argv[2])
    else:
        iterations = 10

    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')

    lastiter = 0
    if len(sys.argv) > 3:
        lastiter = int(sys.argv[3])
    if lastiter > 0:
        model.load_state_dict(torch.load(f'./checkpoints/epoch{lastiter}.pkl'))
        opt.load_state_dict(torch.load(f'./checkpoints/epoch{lastiter}_opt.pkl'))

    # calibs will be used only in forwarding, so we preprocess it into the target device
    # for _, _, _, _, c in trainDataSet:
    #     for k in c.keys():
    #         c[k] = c[k].to(device)

    epochst = time.perf_counter()
    for epoch in range(iterations):
        random.shuffle(trainDataSet)
        clsLossSum, regLossSum = 0.0, 0.0
        maxClsLoss, maxRegLoss = 0.0, 0.0
        clsCnt, regCnt = 0, 0
        if processPool is None:
            def process(data):
                for d in data:
                    yield cputask(d, anchorBevs, gtwithinfo)
            processIter = process(trainDataSet)
        else:
            preprocessed = [processPool.submit(cputask, data, anchorBevs, gtwithinfo) for data in trainDataSet]
            def process(futures):
                for res in as_completed(futures):
                    yield res.result()
            processIter = process(preprocessed)
        for i, res in enumerate(processIter):
            # shape = (N, 35, 9)
            voxel, idx, img, gt, gtbev, pi, ni, gi, calibCpu = res
            calib = {}
            for k in calibCpu.keys():
                calib[k] = torch.Tensor(calibCpu[k]).to(device)

            # shape = (batch, N, 35, 9)
            voxel = voxel[None, :]
            idx = np.concatenate([np.zeros((idx.shape[0], 1)), idx], axis = 1)

            opt.zero_grad()

            with autocast(dtype = cfg.dtype):
                st = time.perf_counter()
                voxel = torch.Tensor(voxel).to(device)
                idx = torch.LongTensor(idx).to(device)
                img = torch.Tensor(img).to(device).permute(2, 0, 1) / 255
                img = img[None, ...]
                if cfg.half:
                    gt = gt.half()
                score, reg = model(voxel, img, idx, [calib], imsize)
                score = score.squeeze(dim = 0).permute(1, 2, 0)
                reg = reg.squeeze(dim = 0).permute(1, 2, 0)
                ed = time.perf_counter()
                forwardTime += ed - st

                l = gt.to(device) if gt is not None else None

                st = time.perf_counter()
                clsLoss, regLoss = criterion(pi, ni, gi, l, score, reg, anchors, 2)
                loss = clsLoss
                if not clsLoss.isnan():
                    clsLossSum += clsLoss.item()
                    maxClsLoss = max(maxClsLoss, clsLoss.item())
                    clsCnt += 1
                if regLoss is not None:
                    loss = loss + regLoss
                    if not regLoss.isnan():
                        regLossSum += regLoss.item()
                        maxRegLoss = max(maxRegLoss, regLoss.item())
                        regCnt += 1
                ed = time.perf_counter()
                lossTime += ed - st

            st = time.perf_counter()
            if cfg.half:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()
            ed = time.perf_counter()
            backwardTime += ed - st

            if (i + 1) % 100 == 0:
                print('\r', forwardTime, lossTime, backwardTime, allTime)

            print(f'\rEpoch{epoch + lastiter + 1} {i + 1}/{len(trainSet)}', end = ' ')
            if (i + 1) % 50 == 0 or i + 1 == len(trainSet):
                print('\nAverage classfication loss: %.6f, Average regression loss: %.6f'
                      % (clsLossSum / (i + 1), regLossSum / regCnt))
                print('Max classfication loss: %.6f, Max regression loss: %.6f'
                      % (maxClsLoss, maxRegLoss))
                # print('Non-nan classfication loss:', clsCnt, 'None-nan regression loss:', regCnt)
            epoched = time.perf_counter()
            allTime = epoched - epochst

        torch.save(model.state_dict(), f'./checkpoints/epoch{epoch + lastiter + 1}.pkl')
        torch.save(opt.state_dict(), f'./checkpoints/epoch{epoch + lastiter + 1}_opt.pkl')

if __name__ == '__main__':
    if cfg.numthreads != -1:
        torch.set_num_threads(cfg.numthreads)
    if cfg.multiprocess > 0:
        with ProcessPoolExecutor(cfg.multiprocess) as pool:
            train(pool)
    else:
        train(None)