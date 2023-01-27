import numpy as np
from typing import Union
import torch

def lidar2P2(pcd: Union[np.ndarray, torch.Tensor], calib: dict):
    """
    Tranform points to P2 \n
    Note that pcd and calib should be the same type in same device
    @param pcd: (N, 3 + C)
    @param calib: Calibration matrices, preprocessed to (4, 4)
    @return: (N, 3)
    """
    assert pcd.ndim == 2, 'Point cloud should be in (N, 3 + C)'
    if isinstance(pcd, np.ndarray):
        points = np.empty((4, pcd.shape[0]), dtype = 'float32')
    elif isinstance(pcd, torch.Tensor):
        points = torch.empty((4, pcd.shape[0]), dtype = torch.float32, device = pcd.device)
    else:
        raise TypeError('pcd should be ndarray or Tensor')
    points[:3] = pcd[:, :3].T
    points[3] = 1
    points = calib['P2'] @ calib['R0_rect'] @ calib['Tr_velo_to_cam'] @ points
    return points[:3].T

def p22Lidar(pcd: Union[np.ndarray, torch.Tensor], calib: dict):
    """
    Inverse of lidar2P2 \n
    Note that pcd and calib should be the same type in same device
    @param pcd: (N, 3 + C)
    @param calib: Calibration matrices, preprocessed to (4, 4)
    @return: (N, 3)
    """
    assert pcd.ndim == 2, 'Point cloud should be in (N, 3 + C)'
    if isinstance(pcd, np.ndarray):
        points = np.empty((4, pcd.shape[0]), dtype = 'float32')
        inverse = np.linalg.inv
    elif isinstance(pcd, torch.Tensor):
        points = torch.empty((4, pcd.shape[0]), dtype = torch.float32, device = pcd.device)
        inverse = torch.linalg.inv
    else:
        raise TypeError('pcd should be ndarray or Tensor')
    points[:3] = pcd[:, :3].T
    points[3] = 1
    points = inverse(calib['Tr_velo_to_cam']) @ inverse(calib['R0_rect']) @ inverse(calib['P2']) @ points
    return points[:3].T

def lidar2Img(pcd: Union[np.ndarray, torch.Tensor], calib: dict, uncheck = False):
    """
    Project points to image \n
    Note that pcd and calib should be the same type in same device
    @param pcd: (N, 3 + C)
    @param calib: Calibration matrices, preprocessed to (4, 4)
    @param uncheck: If True, the function will not discard the points behind the camera
    @return: (N, 2) in (width coor, height coor)
    """
    assert pcd.ndim == 2, 'Point cloud should be in (N, 3 + C)'
    if isinstance(pcd, np.ndarray):
        points = np.empty((4, pcd.shape[0]), dtype = 'float32')
    elif isinstance(pcd, torch.Tensor):
        points = torch.empty((4, pcd.shape[0]), dtype = torch.float32, device = pcd.device)
    else:
        raise TypeError('pcd should be ndarray or Tensor')
    points[:3] = pcd[:, :3].T
    points[3] = 1
    points = calib['R0_rect'] @ calib['Tr_velo_to_cam'] @ points
    if not uncheck:
        points = points[:, points[2] > 0]  # 去掉深度小于0的点（在摄像机后面）
    points = calib['P2'] @ points
    points[:2] = points[:2] / points[2]
    return points[:2].T