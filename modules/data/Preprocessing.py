import numpy as np
import torch
from modules.Extension import cpp
from typing import Sequence, Union, List
from numba import njit
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category = NumbaDeprecationWarning)
warnings.simplefilter('ignore', category = NumbaPendingDeprecationWarning)

def crop(pcd: np.ndarray, range: Sequence[float]):
    low = np.array(range[0:3])
    high = np.array(range[3:6])
    roi = pcd[:, :3]
    f = np.all((low <= roi) & (roi < high), axis = 1)
    return pcd[f]

def cropTensor(pcd: torch.Tensor, range: Sequence[float]):
    low = torch.Tensor(range[0:3]).to(pcd.device)
    high = torch.Tensor(range[3:6]).to(pcd.device)
    roi = pcd[:, :3]
    f = torch.all((low <= roi) & (roi < high), dim = 1) # noqa
    return pcd[f]

def cropToSight(pcd: Union[np.ndarray, torch.Tensor], calib: dict, imsize: Sequence[int]):
    """
    Notice that imsize is in (w, h)
    @param pcd:
    @param calib:
    @param imsize:
    @return:
    """
    # it seems that there exists minor difference between the calculation result from numpy and pytorch
    # so to avoid such inconsistency, the image size is subtracted by a small number
    if isinstance(pcd, np.ndarray):
        imsize = np.array(imsize) - 1e-3
        points = np.empty((4, pcd.shape[0]), dtype = 'float32')
        all = lambda x: np.all(x, axis = 1)
    else:
        imsize = torch.Tensor(imsize).to(pcd.device) - 1e-3
        points = torch.empty((4, pcd.shape[0]), dtype = torch.float32, device = pcd.device)
        all = lambda x: torch.all(x, dim = 1)
    points[:3] = pcd.T[:3]
    points[3] = 1
    points = calib['R0_rect'] @ calib['Tr_velo_to_cam'] @ points
    f = points[2] > 0
    pcd = pcd[f]
    points = points[:, f]  # 去掉深度小于0的点（在摄像机后面）
    points = calib['P2'] @ points
    points[:2] = points[:2] / points[2]
    points = points[:2].T
    f = all(points >= 0) & all(points < imsize) # noqa
    pcd = pcd[f]
    return pcd

def group_(pcd: np.ndarray, range: Sequence[float], size: Sequence[float], samplesPerVoxel: int):
    """
    Group method optimized using cpp
    @param samplesPerVoxel: Sampling times for each voxel
    @param pcd: Point cloud
    @param range: Point cloud range
    @param size: Size of each voxel
    @return: (voxel, indices), voxel in (N, 35, 7), indices in (N, 3)
    """
    np.random.shuffle(pcd)
    pts = pcd[:, :3]
    low = np.array(range[0:3])
    idx = ((pts - low) / size).astype('int32')
    voxel, uidx, vcnt = cpp._group(pcd, idx, samplesPerVoxel) # noqa
    center = voxel[..., :3].sum(axis = 1) / vcnt[:, None]
    voxel[..., 3:6] = voxel[..., :3] - center[:, None, :]
    return voxel, np.array(uidx).T

@njit
def group(pcd: np.ndarray, range: List[float], size: List[float], samplesPerVoxel: int):
    """
    Group method optimized by numba
    @param samplesPerVoxel: Sampling times for each voxel
    @param pcd: Point cloud
    @param range: Point cloud range
    @param size: Size of each voxel
    @return: (voxel, indices), voxel in (N, 35, 9), indices in (N, 3), the last dim contains
            7 encoded information and xy coordinates when points are projected to camera image
    """
    np.random.shuffle(pcd)
    pts = pcd[:, :3]
    low = np.array(range[0:3])
    size = np.array(size)
    idx = ((pts - low) / size).astype(np.int32)
    mp = {}
    cnt = {}
    keys = []
    for p, i in zip(pcd, idx):
        i = (i[0], i[1], i[2])
        if i not in mp:
            mp[i] = np.zeros((samplesPerVoxel, 9))
            cnt[i] = 0
            keys.append(i)
        if cnt[i] < samplesPerVoxel:
            v = mp[i]
            c = cnt[i]
            v[c, 0], v[c, 1], v[c, 2], v[c, 6], v[c, 7], v[c, 8] = p[0], p[1], p[2], p[3], p[4], p[5]
            cnt[i] += 1
    voxel = np.empty((len(mp), samplesPerVoxel, 9))
    vcnt = np.empty(len(mp))
    uidx = np.empty((len(mp), 3))
    for i, k in enumerate(keys):
        voxel[i] = mp[k]
        vcnt[i] = cnt[k]
        uidx[i, 0], uidx[i, 1], uidx[i, 2] = k[0], k[1], k[2]
    vcnt = np.expand_dims(vcnt, axis = 1)
    center = voxel[..., :3].sum(axis = 1) / vcnt
    center = np.expand_dims(center, axis = 1)
    voxel[..., 3:6] = voxel[..., :3] - center
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
