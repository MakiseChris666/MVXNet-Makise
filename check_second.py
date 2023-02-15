import numpy as np
from modules.Calc import decodeRegression, bbox3d2corner, classifyAnchors, bbox3d2bev, bboxLidar2Cam
import torch
import torch.nn.functional as f
import os
import open3d
PointCloud = open3d.geometry.PointCloud
OrientedBoundingBox = open3d.geometry.OrientedBoundingBox
Vector3dVector = open3d.utility.Vector3dVector
from open3d.visualization import draw_geometries
from modules.data.Preprocessing import group, createAnchors
from MVXNet import MVXNet
import modules.config as cfg
from modules.data import Load as load
from modules.utils import lidar2Img
from modules.voxelnet import VoxelLoss, VoxelNet
from modules.nms import nmsbev
from torch.cuda.amp import autocast

device = 'cuda'
dataroot = '/mnt/D/AIProject/mmdetection3d-master/data/kitti'
testInfoPath = os.path.join(dataroot, 'ImageSets/val.txt')
trainInfoPath = os.path.join(dataroot, 'ImageSets/train.txt')

def cputask(data):
    pcd, img, bbox2d, bbox3d, bev, calib = data
    pcd = torch.Tensor(pcd)
    proj = lidar2Img(pcd, calib, True)
    proj = proj[:, [1, 0]]
    pcd = torch.concat([pcd, proj], dim = 1)
    pcd = pcd.numpy()

    voxel, idx = group(pcd, cfg.velorange, cfg.voxelsize, cfg.samplenum)
    return voxel, idx, img, bbox3d, bev, calib

torch.set_num_threads(16)
with open(testInfoPath, 'r') as file:
    testSet = file.read().splitlines()
with open(trainInfoPath, 'r') as file:
    trainSet = file.read().splitlines()

dataset = load.createDataset(trainSet)

anchors = createAnchors(cfg.voxelshape[0] // 2, cfg.voxelshape[1] // 2,
                            cfg.velorange, cfg.carsize)
anchorBevs = bbox3d2bev(anchors.reshape(anchors.shape[:2] + (-1, 7)))
model = VoxelNet()
model.load_state_dict(torch.load('./checkpoints/second/epoch40.pkl'))
criterion = VoxelLoss()

model = model.to(device)
criterion = criterion.to(device)
imsize = torch.Tensor(cfg.imsize).to(device).half()
anchors = anchors.to(device).half()
anchors = anchors.reshape(anchors.shape[0], anchors.shape[1], anchors.shape[2] // 7, 7)
clsLossSum, regLossSum = 0.0, 0.0
maxClsLoss, maxRegLoss = 0.0, 0.0
clsCnt, regCnt = 0, 0

if not os.path.exists('./results'):
    os.mkdir('./results')
if not os.path.exists('./results/data'):
    os.mkdir('./results/data')
resPath = './results/data'

with torch.no_grad():
    for i, data in enumerate(dataset):
        print('\r', i, end = '')
        # shape = (N, 35, 7)
        voxel, idx, img, bbox3d, bev, calib = cputask(data)

        # shape = (batch, N, 35, 7)
        voxel = voxel[None, :]
        voxel = voxel[..., :7]
        idx = np.concatenate([np.zeros((idx.shape[0], 1)), idx], axis = 1)

        with autocast(dtype = cfg.dtype):
            voxel = torch.Tensor(voxel).to(device)
            idx = torch.LongTensor(idx).to(device)
            img = torch.Tensor(img).to(device).permute(2, 0, 1) / 255
            img = img[None, ...]
            score, reg, dir = model(voxel, idx)
            score = score.squeeze(dim = 0).permute(1, 2, 0)
            reg = reg.squeeze(dim = 0).permute(1, 2, 0)
            dir = dir.squeeze(dim = 0).permute(1, 2, 0)

            # if bbox3d is not None:
            #     pi, ni, gi = classifyAnchors(bev, bbox3d[:, :2], anchorBevs, cfg.velorange, 0.45, 0.6)
            #     l = bbox3d.to(device).half()
            # else:
            #     pi, ni, gi = None, None, None
            #     l = None
            #
            # clsLoss, regLoss = criterion(pi, ni, gi, l, score, reg, anchors, 2)
            # loss = clsLoss
            # if not clsLoss.isnan():
            #     clsLossSum += clsLoss.item()
            #     maxClsLoss = max(maxClsLoss, clsLoss.item())
            #     clsCnt += 1
            # if regLoss is not None:
            #     loss = loss + regLoss
            #     if not regLoss.isnan():
            #         regLossSum += regLoss.item()
            #         maxRegLoss = max(maxRegLoss, regLoss.item())
            #         regCnt += 1

            score, reg, dir = score.detach().float(), reg.detach().float(), dir.detach().float()
            reg = reg.reshape(reg.shape[0], reg.shape[1], reg.shape[2] // 7, 7)

        dir = dir.reshape((176, 200, 2, 2))
        dir = f.softmax(dir, dim = -1)
        score = torch.sigmoid(score)
        pos = torch.where(score >= 0.5)
        score = score[pos]
        dir = dir[pos]
        dirneg = dir[..., 0] < dir[..., 1]
        reg = reg[pos]
        reg[:, 6] = torch.clamp(reg[:, 6], -1.0, 1.0)

        bbox = decodeRegression(reg, anchors[pos])
        bbox[bbox[:, 6] <= 0, 6] += torch.pi
        bbox[dirneg, 6] -= torch.pi

        score = score.cpu()
        bbox = bbox.cpu()
        reserved = nmsbev(score, bbox)
        bbox = bbox[reserved]
        score = score[reserved]
        bbox = bbox[torch.logical_not(torch.any(bbox.isnan(), dim = 1))]
        # bbox = bbox[torch.all(bbox[:, 3:6] < 10, dim = 1)]
        # bbox = bbox[torch.all(bbox[:, 3:6] > 0.1, dim = 1)]
        corners = bbox3d2corner(bbox).cpu().type(torch.float32)
        corners = corners.reshape((-1, 3))
        corners2d = lidar2Img(corners, calib, True)
        corners2d = corners2d.reshape((-1, 8, 2))
        bbox2d = torch.concat([torch.min(corners2d, dim = 1, keepdim = True)[0], torch.max(corners2d, dim = 1, keepdim = True)[0]], dim = 1)
        bbox2d = bbox2d.reshape((-1, 4))

        # o3dBox = []
        # for c in corners:
        #     o3dBox.append(OrientedBoundingBox().create_from_points(Vector3dVector(c)))
        # for b in o3dBox:
        #     b.color = (1, 0, 0)
        # pcd = PointCloud(Vector3dVector(data[0][:, :3]))
        # draw_geometries([pcd, *o3dBox])

        bbox = bboxLidar2Cam(bbox, calib['Tr_velo_to_cam'], True)
        bbox = bbox.numpy().tolist()
        bbox2d = bbox2d.numpy().tolist()
        path = os.path.join(resPath, trainSet[i] + '.txt')
        with open(path, 'w') as file:
            for res2d, res3d, s in zip(bbox2d, bbox, score):
                print('Car 0.0 0 0.0 %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' %
                      tuple(res2d + res3d + [s]), file = file)

# print('Avg classfication loss: %.6f, Avg regression loss: %.6f'
#       % (clsLossSum / clsCnt, regLossSum / regCnt))
# print('Max classfication loss: %.6f, Max regression loss: %.6f'
#       % (maxClsLoss, maxRegLoss))