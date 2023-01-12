import os
import sys
import time

import numpy as np
import torch
from modules.voxelnet import Pipe, Loss
from torch import nn
from torch.autograd import Variable

from modules import Calc
from modules import Config as cfg
from modules.data import Load
from modules.data import Preprocessing as pre


def initWeights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()

class VoxelNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.svfe = Pipe.SVFE(cfg.samplenum)
        self.fcn = Pipe.FCN(128, 128)
        self.cml = Pipe.CML()
        self.rpn = Pipe.RPN()

    @staticmethod
    def reindex(x, idx):
        # input shape: x = (batch * N, 128), idx = (batch * N, 1 + 3)
        res = Variable(torch.Tensor(1, 128, cfg.voxelshape[2], cfg.voxelshape[0]
                                    , cfg.voxelshape[1]).to(device))
        res[idx[:, 0], :, idx[:, 3], idx[:, 1], idx[:, 2]] = x
        return res

    def forward(self, x, idx):
        # x shape = (batch, N, 35, 7), idx shape = (batch * N, 1 + 3), idx[:, 0] is the batch No.
        # idx: corresponding indices to voxels
        x = self.svfe(x)
        x = self.fcn(x)
        # after SVFE&FCN: shape = (batch, N, 35, 128)
        x = torch.max(x, dim = 2)[0]
        # after elementwise max: shape = (batch, N, 1, 128)
        x = torch.squeeze(x, dim = 2).reshape((-1, 128))
        # shape = (batch * N, 128)
        x = self.reindex(x, idx)
        x = self.cml(x)
        x = x.reshape((1, -1, cfg.voxelshape[0], cfg.voxelshape[1]))
        score, reg = self.rpn(x)
        return score, reg

device = cfg.device
dataroot = '../mmdetection3d-master/data/kitti'
if len(sys.argv) > 1 and sys.argv[1] != '#':
    dataroot = sys.argv[1]
trainInfoPath = os.path.join(dataroot, 'ImageSets/train.txt')
testInfoPath = os.path.join(dataroot, 'ImageSets/val.txt')

if cfg.numthreads != -1:
    torch.set_num_threads(cfg.numthreads)

def train():

    with open(trainInfoPath, 'r') as f:
        trainSet = f.read().splitlines()
    with open(testInfoPath, 'r') as f:
        testSet = f.read().splitlines()

    trainX, trainY = Load.createDataset(trainSet)
    testX, testY = Load.createDataset(testSet)

    anchors = pre.createAnchors(cfg.voxelshape[0] // 2, cfg.voxelshape[1] // 2,
                                          cfg.velorange, cfg.carsize)
    model = VoxelNet()
    anchorBevs = Calc.bbox3d2bev(anchors)
    # anchorPolygons = Calc.getPolygons(anchorBevs).reshape((cfg.voxelshape[0] // 2, cfg.voxelshape[1] // 2, 2))
    criterion = Loss.VoxelLoss()
    opt = torch.optim.Adam(model.parameters(), lr = 0.001)
    # torch.autograd.set_detect_anomaly(True)

    model = model.to(device)
    anchors = anchors.to(device)
    criterion = criterion.to(device)
    model.apply(initWeights)

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

    epochst = time.perf_counter()
    for epoch in range(iterations):
        for i, (x, y) in enumerate(zip(trainX, trainY)):
            # shape = (N, 35, 7)
            st = time.perf_counter()
            voxel, idx = pre.group(x, cfg.velorange, cfg.voxelsize, cfg.samplenum)
            # voxel_, idx_ = pre.group_(x, cfg.velorange, cfg.voxelsize, 25)

            ed = time.perf_counter()
            groupTime += ed - st
            # shape = (batch, N, 35, 7)
            voxel = voxel[None, :]
            idx = np.concatenate([np.zeros((idx.shape[0], 1)), idx], axis = 1)

            opt.zero_grad()
            st = time.perf_counter()
            voxel = torch.Tensor(voxel).to(device)
            idx = torch.LongTensor(idx).to(device)
            score, reg = model(voxel, idx)
            score = score.squeeze(dim = 0).permute(1, 2, 0)
            reg = reg.squeeze(dim = 0).permute(1, 2, 0)
            ed = time.perf_counter()
            forwardTime += ed - st

            st = time.perf_counter()
            if y[0] is not None:
                # gtPolygons = Calc.getPolygons(y[1])
                # pi, ni, gi = Calc.classifyAnchors_(gtPolygons, y[0][:, [0, 1]], anchorPolygons, cfg.velorange, 0.45, 0.6)

                pi, ni, gi = Calc.classifyAnchors(y[1], y[0][:, [0, 1]], anchorBevs, cfg.velorange, 0.45, 0.6)

                # pi = pi.cuda()
                # ni = ni.cuda()
                # gi = gi.cuda()
                l = y[0].to(device)
            else:
                pi, ni, gi, l = None, None, None, None
                # pos, neg, gi, pi, l = None, None, None, None, None
            ed = time.perf_counter()
            classifyTime += ed - st

            st = time.perf_counter()
            clsLoss, regLoss = criterion(pi, ni, gi, l, score, reg, anchors, 2)
            loss = clsLoss
            if regLoss is not None:
                loss = loss + regLoss
            ed = time.perf_counter()
            lossTime += ed - st

            st = time.perf_counter()
            loss.backward()
            opt.step()
            ed = time.perf_counter()
            backwardTime += ed - st

            if (i + 1) % 50 == 0:
                print('\r', groupTime, forwardTime, classifyTime, lossTime, backwardTime, allTime)

            print(f'\rEpoch{epoch + lastiter + 1} {i + 1}/{len(trainSet)}', 'Classification Loss:', clsLoss.item()
                  , 'Regression Loss:', 'None' if regLoss is None else regLoss.item(), end = '')
            epoched = time.perf_counter()
            allTime = epoched - epochst

        torch.save(model.state_dict(), f'./checkpoints/epoch{epoch + lastiter + 1}.pkl')