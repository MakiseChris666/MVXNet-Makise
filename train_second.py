import random
import modules.config as cfg
import torch
import os
import time
import contextlib
from modules.config import options, trainInfoPath
from modules.Calc import bbox3d2bev, bbox3d2corner, classifyAnchors, classifyAnchorsAlignedGT, bbox3dAxisAlign
from modules.data import Load as load, Preprocessing as pre
from modules.augment.Augment import augmentTargetClasses
from modules.augment.LoadGT import getAllGT
from modules.utils import lidar2Img
from MVXNet import MVXNet
from modules.voxelnet import VoxelLoss, VoxelNet
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch.cuda.amp import autocast
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad_norm_
# from open3d.cuda.pybind.utility import Vector3dVector
# from open3d.cuda.pybind.geometry import PointCloud, OrientedBoundingBox
# from open3d.visualization import draw_geometries

device = cfg.device
if cfg.half:
    from torch.cuda.amp import GradScaler
    scaler = GradScaler()

if cfg.sparsemiddle:
    try:
        import spconv.pytorch as spconv
    except ImportError:
        raise ImportError('Please install spconv first to use sparse middle layers.')

def initWeights(m):
    if isinstance(m, nn.Conv2d) and m.get_parameter('weight').requires_grad:
        nn.init.kaiming_normal_(m.weight.data, mode = 'fan_out', nonlinearity = 'relu')
        m.bias.data.zero_()
    # elif isinstance(m, nn.Linear) and m.get_parameter('weight').requires_grad:
    #     nn.init.kaiming_normal_(m.weight.data, nonlinearity = 'relu')

def cputask1(data, anchorsXYXY, gtwithinfo):
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
    # pi, ni, gi = classifyAnchors(bev, bbox3d[:, [0, 1]], anchorBevs, cfg.velorange, 0.45, 0.6)
    bbox3dXYXY = bbox3dAxisAlign(bbox3d)
    pi, ni, gi = classifyAnchorsAlignedGT(bbox3dXYXY, anchorsXYXY, 0.45, 0.6)
    return voxel, idx, img, bbox3d, bev, pi, ni, gi, calib

def cputask2(data, anchorsXYXY, gtwithinfo):
    pcd, img, bbox2d, bbox3d, bev, calib = data
    pcd = torch.Tensor(pcd)
    proj = lidar2Img(pcd, calib, True)
    proj = proj[:, [1, 0]]
    pcd = torch.concat([pcd, proj], dim = 1)
    pcd = pcd.numpy()

    voxel, idx = pre.group(pcd, cfg.velorange, cfg.voxelsize, cfg.samplenum)
    # pi, ni, gi = classifyAnchors(bev, bbox3d[:, [0, 1]], anchorBevs, cfg.velorange, 0.45, 0.6)
    bbox3dXYXY = bbox3dAxisAlign(bbox3d)
    pi, ni, gi = classifyAnchorsAlignedGT(bbox3dXYXY, anchorsXYXY, 0.45, 0.6)
    return voxel, idx, img, bbox3d, bev, pi, ni, gi, calib

cputask = cputask1 if cfg.augment else cputask2

def train(processPool):

    with open(trainInfoPath, 'r') as f:
        trainSet = f.read().splitlines()

    trainDataSet = load.createDataset(trainSet)
    gtwithinfo = getAllGT(['Car']) if cfg.augment else None

    anchors = pre.createAnchors(cfg.voxelshape[0] // 2, cfg.voxelshape[1] // 2,
                                          cfg.velorange, cfg.carsize)
    # anchorBevs = bbox3d2bev(anchors.reshape(anchors.shape[:2] + (-1, 7)))
    anchorsXYXY = bbox3dAxisAlign(anchors.reshape(anchors.shape[:2] + (-1, 7)))
    # model = MVXNet()
    # model.backbone.apply(initWeights)
    # model.head.fusion.apply(initWeights)
    model = VoxelNet()
    model.apply(initWeights)
    criterion = VoxelLoss()
    trainParams = filter(lambda p: p.requires_grad, model.parameters())
    opt = torch.optim.AdamW(trainParams, eps = 1e-3 if cfg.half else 1e-8,
                            lr = 1e-3, weight_decay = 0.01)
    for p in opt.param_groups:
        p['initial_lr'] = 1e-4
        p['max_lr'] = 3e-3
        p['min_lr'] = 1e-5
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

    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    if not os.path.exists('./checkpoints/second'):
        os.mkdir('./checkpoints/second')

    iterations = options.numepochs
    lastiter = options.lastiter
    scheduler = OneCycleLR(opt, max_lr = 3e-3, div_factor = 30, final_div_factor = 10, epochs = 40, steps_per_epoch = 3712,
                           last_epoch = lastiter * 3712 - 1, cycle_momentum = False, pct_start = 0.05)
    if lastiter > 0:
        model.load_state_dict(torch.load(f'./checkpoints/second/epoch{lastiter}.pkl'))
        opt.load_state_dict(torch.load(f'./checkpoints/second/epoch{lastiter}_opt.pkl'))

    # calibs will be used only in forwarding, so we preprocess it into the target device
    # for _, _, _, _, c in trainDataSet:
    #     for k in c.keys():
    #         c[k] = c[k].to(device)
    cast = autocast() if cfg.half else contextlib.nullcontext()

    epochst = time.perf_counter()
    for epoch in range(iterations):
        random.shuffle(trainDataSet)
        clsLossSum, regLossSum, dirLossSum = 0.0, 0.0, 0.0
        maxClsLoss, maxRegLoss, maxDirLoss = 0.0, 0.0, 0.0
        clsCnt, regCnt, dirCnt = 0, 0, 0
        if processPool is None:
            def process(data):
                for d in data:
                    yield cputask(d, anchorsXYXY, gtwithinfo)
            processIter = process(trainDataSet)
        else:
            preprocessed = [processPool.submit(cputask, data, anchorsXYXY, gtwithinfo) for data in trainDataSet]
            def process(futures):
                for res in as_completed(futures):
                    yield res.result()
            processIter = process(preprocessed)
        for i, res in enumerate(processIter):
            # shape = (N, 35, 9)
            voxel, idx, img, gt, gtbev, pi, ni, gi, calibCpu = res
            # print(pi[0].shape[0])
            # continue
            calib = {}
            for k in calibCpu.keys():
                calib[k] = torch.Tensor(calibCpu[k]).to(device)

            # shape = (batch, N, 35, 9)
            voxel = voxel[None, :]
            voxel = voxel[..., :7]
            idx = np.concatenate([np.zeros((idx.shape[0], 1)), idx], axis = 1)

            opt.zero_grad()

            with cast:
                st = time.perf_counter()
                voxel = torch.Tensor(voxel).to(device)
                idx = torch.LongTensor(idx).to(device)
                img = torch.Tensor(img).to(device).permute(2, 0, 1) / 255
                img = img[None, ...]
                if cfg.half:
                    gt = gt.half()
                score, reg, dir = model(voxel, idx)
                score = score.squeeze(dim = 0).permute(1, 2, 0)
                reg = reg.squeeze(dim = 0).permute(1, 2, 0)
                dir = dir.squeeze(dim = 0).permute(1, 2, 0)
                ed = time.perf_counter()
                forwardTime += ed - st

                l = gt.to(device) if gt is not None else None

                st = time.perf_counter()
                clsLoss, regLoss, dirLoss = criterion(pi, ni, gi, l, score, reg, dir, anchors, 2)
                loss = clsLoss
                if not clsLoss.isnan():
                    clsLossSum += clsLoss.item()
                    maxClsLoss = max(maxClsLoss, clsLoss.item())
                    clsCnt += 1
                if regLoss is not None:
                    loss = loss + regLoss + dirLoss
                    if not regLoss.isnan():
                        regLossSum += regLoss.item()
                        maxRegLoss = max(maxRegLoss, regLoss.item())
                        regCnt += 1
                    if not dirLoss.isnan():
                        dirLossSum += dirLoss.item()
                        maxDirLoss = max(maxDirLoss, dirLoss.item())
                        dirCnt += 1
                ed = time.perf_counter()
                lossTime += ed - st

            st = time.perf_counter()
            if cfg.half:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                clip_grad_norm_(trainParams, 35)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                clip_grad_norm_(trainParams, 35)
                opt.step()
            ed = time.perf_counter()
            backwardTime += ed - st

            scheduler.step()
            if (i + 1) % 100 == 0:
                epoched = time.perf_counter()
                allTime = epoched - epochst
                print('\r', forwardTime, lossTime, backwardTime, allTime)
                print('lr =', scheduler.get_last_lr())

            print(f'\rEpoch{epoch + lastiter + 1} {i + 1}/{len(trainSet)}', end = ' ')
            # print(clsLoss.item(), regLoss.item(), dirLoss.item())
            if (i + 1) % 50 == 0 or i + 1 == len(trainSet):
                print()
                print('Avg classfication loss: %.7f, Avg regression loss: %.7f, Avg direction loss: %.7f'
                      % (clsLossSum / clsCnt, regLossSum / regCnt, dirLossSum / dirCnt))
                print('Max classfication loss: %.7f, Max regression loss: %.7f, Max direction loss: %.7f'
                      % (maxClsLoss, maxRegLoss, maxDirLoss))
                if i + 1 == len(trainSet):
                    with open('./checkpoints/second/log.txt', 'a+') as logf:
                        print(f'Epoch {epoch + lastiter + 1}:', file = logf)
                        print('Avg classfication loss: %.7f, Avg regression loss: %.7f, Avg direction loss: %.7f'
                              % (clsLossSum / clsCnt, regLossSum / regCnt, dirLossSum / dirCnt), file = logf)
                        print('Max classfication loss: %.7f, Max regression loss: %.7f, Max direction loss: %.7f'
                              % (maxClsLoss, maxRegLoss, maxDirLoss), file = logf)
                # print('Non-nan classfication loss:', clsCnt, 'None-nan regression loss:', regCnt)

        torch.save(model.state_dict(), f'./checkpoints/second/epoch{epoch + lastiter + 1}.pkl')
        torch.save(opt.state_dict(), f'./checkpoints/second/epoch{epoch + lastiter + 1}_opt.pkl')

if __name__ == '__main__':
    if cfg.numthreads != -1:
        torch.set_num_threads(cfg.numthreads)
    if cfg.multiprocess > 0:
        with ProcessPoolExecutor(cfg.multiprocess) as pool:
            train(pool)
    else:
        train(None)