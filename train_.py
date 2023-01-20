import random
from modules import Config as cfg
import sys
import torch
import os
import time
from modules.Calc import bbox3d2bev, classifyAnchors
from modules.data import Load as load, Preprocessing as pre
from MVXNet import MVXNet
from modules.voxelnet import VoxelLoss
import numpy as np

device = cfg.device
dataroot = '../mmdetection3d-master/data/kitti'
if len(sys.argv) > 1 and sys.argv[1] != '-':
    dataroot = sys.argv[1]
trainInfoPath = os.path.join(dataroot, 'ImageSets/train.txt')
testInfoPath = os.path.join(dataroot, 'ImageSets/val.txt')

def trainSingle():

    with open(trainInfoPath, 'r') as f:
        trainSet = f.read().splitlines()

    trainDataSet = load.createDataset(trainSet)

    anchors = pre.createAnchors(cfg.voxelshape[0] // 2, cfg.voxelshape[1] // 2,
                                          cfg.velorange, cfg.carsize)
    anchorBevs = bbox3d2bev(anchors.reshape(anchors.shape[:2] + (-1, 7)))
    model = MVXNet()
    criterion = VoxelLoss()
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr = 0.001)
    # torch.autograd.set_detect_anomaly(True)

    model = model.to(device)
    anchors = anchors.to(device)
    criterion = criterion.to(device)
    imsize = torch.Tensor(cfg.imsize).to(device)

    groupTime = 0
    forwardTime = 0
    classifyTime = 0
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

    # calibs will be used only in forwarding, so we preprocess it into the target device
    for _, _, _, _, c in trainDataSet:
        for k in c.keys():
            c[k] = c[k].to(device)

    epochst = time.perf_counter()
    for epoch in range(iterations):
        random.shuffle(trainDataSet)
        clsLossSum, regLossSum = 0.0, 0.0
        maxClsLoss, maxRegLoss = 0.0, 0.0
        regCnt = 0
        for i, (pcd, img, gt, gtbev, calib) in enumerate(trainDataSet):
            # shape = (N, 35, 7)
            st = time.perf_counter()
            voxel, idx = pre.group(pcd, cfg.velorange, cfg.voxelsize, cfg.samplenum)

            ed = time.perf_counter()
            groupTime += ed - st
            # shape = (batch, N, 35, 7)
            voxel = voxel[None, :]
            idx = np.concatenate([np.zeros((idx.shape[0], 1)), idx], axis = 1)

            opt.zero_grad()
            st = time.perf_counter()
            voxel = torch.Tensor(voxel).to(device)
            idx = torch.LongTensor(idx).to(device)
            img = torch.Tensor(img).to(device).permute(2, 0, 1) / 255
            img = img[None, ...]
            score, reg = model(voxel, img, idx, [calib], imsize)
            score = score.squeeze(dim = 0).permute(1, 2, 0)
            reg = reg.squeeze(dim = 0).permute(1, 2, 0)
            ed = time.perf_counter()
            forwardTime += ed - st

            st = time.perf_counter()
            if gt is not None:
                pi, ni, gi = classifyAnchors(gtbev, gt[:, [0, 1]], anchorBevs, cfg.velorange, 0.45, 0.6)
                l = gt.to(device)
            else:
                pi, ni, gi, l = None, None, None, None

            ed = time.perf_counter()
            classifyTime += ed - st

            st = time.perf_counter()
            clsLoss, regLoss = criterion(pi, ni, gi, l, score, reg, anchors, 2)
            loss = clsLoss
            clsLossSum += clsLoss.item()
            maxClsLoss = max(maxClsLoss, clsLoss.item())
            if regLoss is not None:
                loss = loss + regLoss
                regLossSum += regLoss.item()
                maxRegLoss = max(maxRegLoss, regLoss.item())
                regCnt += 1
            ed = time.perf_counter()
            lossTime += ed - st

            st = time.perf_counter()
            loss.backward()
            opt.step()
            ed = time.perf_counter()
            backwardTime += ed - st

            if (i + 1) % 100 == 0:
                print('\r', groupTime, forwardTime, classifyTime, lossTime, backwardTime, allTime)

            print(f'\rEpoch{epoch + lastiter + 1} {i + 1}/{len(trainSet)}', end = ' ')
            if (i + 1) % 50 == 0 or i + 1 == len(trainSet):
                print('\nAverage classfication loss: %.6f, Average regression loss: %.6f'
                      % (clsLossSum / (i + 1), regLossSum / regCnt))
                print('Max classfication loss: %.6f, Max regression loss: %.6f'
                      % (maxClsLoss, maxRegLoss))
            epoched = time.perf_counter()
            allTime = epoched - epochst

        torch.save(model.state_dict(), f'./checkpoints/epoch{epoch + lastiter + 1}.pkl')

# if __name__ == '__main__':
#     if cfg.numthreads != -1:
#         torch.set_num_threads(cfg.numthreads)
#     train()