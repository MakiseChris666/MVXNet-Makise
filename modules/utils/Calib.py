import numpy as np

def lidar2CamP2(velo: np.ndarray, calib: dict, discardUnderZero = True) -> np.ndarray:
    """
    Transfer point cloud from LiDAR coordinates to camera coordinates
    @param velo: Point cloud in (4, N)
    @param calib: Calibration, the matrix in it should be preprocessed to (4, 4)
    @param discardUnderZero: Whether to discard the points with z coordinate under zero, i.e. behind the camera
    @return: Point cloud in camera coordinates
    """
    assert velo.ndim == 2, 'Point cloud should be in (3, N)'
    velo = calib['R0_rect'] @ calib['Tr_velo_to_cam'] @ velo
    if discardUnderZero:
        velo = velo[:, velo[2] >= 0]  # 去掉深度小于0的点（在摄像机后面）
    velo = calib['P2'] @ velo
    return velo

def lidar2Img(points, calib: dict):
    """
    Project points to image
    @param points: (N, 3)
    @param calib: Calibration matrices
    @return: (N, 2) in (width coor, height coor)
    """
    assert points.ndim == 2, 'Point cloud should be in (N, 3)'
    points = points.T
    points = calib['R0_rect'] @ calib['Tr_velo_to_cam'] @ points
    points = points[:, points[2] >= 0]  # 去掉深度小于0的点（在摄像机后面）
    points = calib['P2'] @ points
    points[:2] = points[:2] / points[2]
    return points[:2].T