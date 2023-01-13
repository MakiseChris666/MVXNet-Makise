import numpy as np
import torch
from modules.Extension import cpp
from modules import utils
from typing import Sequence

def crop(pcd: np.ndarray, range: Sequence[float]):
    low = np.array(range[0:3])
    high = np.array(range[3:6])
    roi = pcd[:, :3]
    f = np.all((low <= roi) & (roi < high), axis = 1)
    return pcd[f]

def cropToSight(pcd: np.ndarray, calib: dict, imsize: Sequence[int]):
    """
    Notice that imsize is in (w, h)
    @param pcd:
    @param calib:
    @param imsize:
    @return:
    """
    imsize = np.array(imsize)
    points = pcd.T[:3]
    points = np.concatenate([points, np.ones((1, points.shape[1]))], axis = 0)
    points = calib['R0_rect'] @ calib['Tr_velo_to_cam'] @ points
    f = points[2] > 0
    pcd = pcd[f]
    points = points[:, f]  # 去掉深度小于0的点（在摄像机后面）
    points = calib['P2'] @ points
    points[:2] = points[:2] / points[2]
    points = points[:2].T
    f = (points >= 0).all(axis = 1) & (points < imsize).all(axis = 1)
    pcd = pcd[f]
    # img = utils.lidar2Img(pcd, calib, True)
    # print(np.max(img, axis = 0))
    return pcd

def group(pcd: np.ndarray, range: Sequence[float], size, samplesPerVoxel: int):
    np.random.shuffle(pcd)
    pts = pcd[:, :3]
    low = np.array(range[0:3])
    idx = ((pts - low) / size).astype('int32')
    voxel, uidx, vcnt = cpp._group(pcd, idx, samplesPerVoxel) # noqa
    center = voxel[..., :3].sum(axis = 1) / vcnt[:, None]
    voxel[..., 3:6] = voxel[..., :3] - center[:, None, :]
    return voxel, np.array(uidx).T

def group_(pcd, range, size, samplesPerVoxel):
    """

    @param samplesPerVoxel:
    @param pcd:
    @param range:
    @param size:
    @return: (voxel, indices), voxel in (N, 35, 7), indices in (N, 3)
    """
    np.random.shuffle(pcd)
    pts = pcd[:, :3]
    low = np.array(range[0:3])
    idx = ((pts - low) / size).astype('int32')
    uidx, inv = np.unique(idx, axis = 0, return_inverse = True)
    voxel = np.zeros((len(uidx), samplesPerVoxel, 7))
    vcnt = np.zeros(len(uidx), dtype = 'int32')
    for i, p in zip(inv, pcd):
        if vcnt[i] == samplesPerVoxel:
            continue
        voxel[i, vcnt[i], [0, 1, 2, 6]] = p
        vcnt[i] += 1
    center = voxel[..., :3].sum(axis = 1) / vcnt[:, None]
    voxel[..., 3:6] = voxel[..., :3] - center[:, None, :]
    zero = (voxel[..., :3] == 0).all(axis = 2)
    zero = np.where(zero)
    voxel[zero[0], zero[1], 3:6] = 0
    return voxel, uidx

def createAnchors(l, w, range, size):
    """

    @param l:
    @param w:
    @param range:
    @param size:
    @return: (l, w, 14)
    """
    ls = (range[3] - range[0]) / l
    ws = (range[4] - range[1]) / w
    x = torch.linspace(range[0] + ls / 2, range[3] - ls / 2, l)
    y = torch.linspace(range[1] + ws / 2, range[4] - ws / 2, w)
    x, y = torch.meshgrid(x, y, indexing = 'ij')
    g = torch.concat([x[..., None], y[..., None]], dim = 2)
    size = torch.Tensor(size)
    size = torch.tile(size, (l, w, 1))
    z = torch.empty((l, w, 1))
    z[...] = -1
    t = torch.zeros((l, w, 1))
    t2 = torch.empty((l, w, 1))
    t2[...] = torch.pi / 2
    anchors = torch.concat([g, z, size, t], dim = 2)
    anchors2 = torch.concat([g, z, size, t2], dim = 2)
    return torch.concat([anchors, anchors2], dim = 2)
