from shapely.geometry import Polygon
import torch
import numpy as np
from .Extension import cpp
from typing import Sequence, Tuple, Union
from torchvision.ops.boxes import box_iou

index3d = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
emptyIndex3d = (torch.LongTensor([]), torch.LongTensor([]), torch.LongTensor([]))

def getRotationMatrices(r: torch.Tensor):
    rcos = torch.cos(r).reshape((-1, 1))
    rsin = torch.sin(r).reshape((-1, 1))
    rot = torch.concat([rcos, -rsin, rsin, rcos], dim = 1).reshape((-1, 2, 2))
    return rot

def bbox3d2bev(bbox3ds: torch.Tensor) -> torch.Tensor:
    """
    Convert 3d bboxes to bev(in format of corner points)
    @param bbox3ds: (..., 7) in xyzlwhr format, '...' can be none in case of a single box
    @return: bev in (..., 4, 2)
    """
    assert bbox3ds.shape[-1] >= 7
    origshape = bbox3ds.shape[:-1]
    bbox3ds = bbox3ds.reshape((-1, bbox3ds.shape[-1]))
    res = torch.Tensor([[0.5, 0.5], [-0.5, 0.5], [-0.5, -0.5], [0.5, -0.5]]).to(bbox3ds.device)
    res = torch.tile(res, (bbox3ds.shape[0], 1, 1))
    res = res * bbox3ds[:, None, [3, 4]]
    # res shape = (N, 4, 2)
    r = bbox3ds[:, 6]
    rot = getRotationMatrices(r)
    # rot shape = (N, 2, 2)
    res = res @ rot
    res = res + bbox3ds[:, None, [0, 1]]
    if len(origshape) > 0:
        res = res.reshape(origshape + (4, 2))
    else:
        res = res[0]
    return res

def bbox3d2corner(bbox3ds: torch.Tensor) -> torch.Tensor:
    """
    Convert 3d bboxes to corners
    @param bbox3ds: (..., 7) in xyzlwhr format, '...' can be none in case of a single box
    @return: corners of each 3d box in (..., 8, 3)
    """
    assert bbox3ds.shape[-1] >= 7
    origshape = bbox3ds.shape[:-1]
    bbox3ds = bbox3ds.reshape((-1, bbox3ds.shape[-1]))
    bevs = bbox3d2bev(bbox3ds)
    h = bbox3ds[..., [5]]
    z = bbox3ds[..., [2]]
    h = torch.tile(h[..., None], (1, 4, 1))
    z = torch.tile(z[..., None], (1, 4, 1))
    top = torch.concat([bevs, z + h], dim = 2)
    bot = torch.concat([bevs, z], dim = 2)
    res = torch.concat([top, bot], dim = 1)
    if len(origshape) > 0:
        res = res.reshape(origshape + (8, 3))
    else:
        res = res[0]
    return res

def bbox3dAxisAlign(bbox3ds: torch.Tensor) -> torch.Tensor:
    r = torch.abs(bbox3ds[..., 6])
    horizontal = ((r > torch.pi / 4) & (r < torch.pi * 3 / 4))[..., None]
    bbox2d = bbox3ds[..., [0, 1, 3, 4]]
    bbox2d = torch.where(horizontal, bbox2d[..., [0, 1, 3, 2]], bbox2d)
    res = torch.empty_like(bbox2d)
    res[..., :2] = bbox2d[..., :2] - bbox2d[..., 2:] / 2
    res[..., 2:] = bbox2d[..., :2] + bbox2d[..., 2:] / 2
    return res

def iou2d(rs1: Sequence[Polygon], rs2: Sequence[Polygon]):
    """
    Compute pairwise iou between every box in rs1 and rs2
    @param rs1: (N, 4, 2) in corner points format
    @param rs2: (M, 4, 2) in corner points format
    @return: (N, M),
    """
    res = torch.empty((len(rs1), len(rs2)))
    for i, r1 in enumerate(rs1):
        for j, r2 in enumerate(rs2):
            inter = r1.intersection(r2).area
            iou = inter / (r1.area + r2.area - inter)
            res[i, j] = iou
    return res

def getPolygons(bboxes):
    """
    Get a numpy array of shapely.geometry.Polygons
    @param bboxes: (N, 4, 2)
    @return:
    """
    res = np.empty(bboxes.shape[0], dtype = 'object')
    for i, p in enumerate(bboxes):
        res[i] = Polygon(p)
    return res

def classifyAnchorsAlignedGT(gtsXYXY: torch.Tensor, anchorsXYXY: torch.Tensor, negThr: float, posThr: float):
    assert gtsXYXY.shape[-1] == anchorsXYXY.shape[-1] and gtsXYXY.shape[-1] == 4
    anchorShape = anchorsXYXY.shape
    anchorsXYXY = anchorsXYXY.reshape((-1, 4))
    ious = box_iou(anchorsXYXY, gtsXYXY)
    maxious, indices = torch.max(ious, dim = 1)
    pos = maxious > posThr
    nonneg = maxious > negThr
    gi = indices[pos]
    pi = torch.where(pos.reshape(anchorShape[:-1]))
    ni = torch.where(nonneg.reshape(anchorShape[:-1]))
    return pi, ni, gi

def classifyAnchors(gts: torch.Tensor, gtCenters: torch.Tensor, anchors: torch.Tensor,
                    velorange: Sequence[float], negThr: float, posThr: float)\
                    -> Tuple[index3d, index3d, torch.Tensor]:
    if gts is None:
        return emptyIndex3d, emptyIndex3d, torch.LongTensor([])
    l = (velorange[3] - velorange[0]) / anchors.shape[0]
    w = (velorange[4] - velorange[1]) / anchors.shape[1]
    nls = ((gtCenters[:, 0] - velorange[0] - l / 2) / l + 0.5).long()
    nws = ((gtCenters[:, 1] - velorange[1] - w / 2) / w + 0.5).long()
    pi, ni, gi = cpp._classifyAnchors(gts, anchors, nls, nws, negThr, posThr) # noqa
    return pi, ni, gi

def classifyAnchors_(gts, gtCenters, anchors, velorange, negThr, posThr):
    """

    @param gtCenters:
    @param velorange:
    @param gts: Preprocessed by getPolygons
    @param anchors: Preprocessed by getPolygons
    @param negThr:
    @param posThr:
    @return: (pos, neg, gi), gi is the corresponding indices of gt to every positive anchor
    """
    pos = torch.zeros(anchors.shape, dtype = torch.bool)
    neg = torch.ones(anchors.shape, dtype = torch.bool)
    gi = torch.zeros_like(pos, dtype = torch.int64)
    anchorsPerLoc = anchors.shape[2]
    l = (velorange[3] - velorange[0]) / anchors.shape[0]
    w = (velorange[4] - velorange[1]) / anchors.shape[1]
    nls = ((gtCenters[:, 0] - velorange[0] - l / 2) / l + 0.5).long()
    nws = ((gtCenters[:, 1] - velorange[1] - w / 2) / w + 0.5).long()
    anchorArea = anchors[0, 0, 0].area
    for i, gt in enumerate(gts):
        nl = nls[i]
        nw = nws[i]
        gtArea = gt.area
        for z in range(anchorsPerLoc):
            h = 0
            while nl + h < anchors.shape[0]:
                inter = anchors[nl + h, nw, z].intersection(gt).area
                iou = inter / (gtArea + anchorArea - inter)
                if iou < 0.1:
                    break
                if iou >= posThr:
                    pos[nl + h, nw, z] = 1
                    gi[nl + h, nw, z] = i
                    neg[nl + h, nw, z] = 0
                elif iou >= negThr:
                    neg[nl + h, nw, z] = 0
                v = 1
                while nw + v < anchors.shape[1]:
                    inter = anchors[nl + h, nw + v, z].intersection(gt).area
                    iou = inter / (gtArea + anchorArea - inter)
                    if iou < 0.1:
                        break
                    if iou >= posThr:
                        pos[nl + h, nw + v, z] = 1
                        gi[nl + h, nw + v, z] = i
                        neg[nl + h, nw + v, z] = 0
                    elif iou >= negThr:
                        neg[nl + h, nw + v, z] = 0
                    v += 1
                v = -1
                while nw + v >= 0:
                    inter = anchors[nl + h, nw + v, z].intersection(gt).area
                    iou = inter / (gtArea + anchorArea - inter)
                    if iou < 0.1:
                        break
                    if iou >= posThr:
                        pos[nl + h, nw + v, z] = 1
                        gi[nl + h, nw + v, z] = i
                        neg[nl + h, nw + v, z] = 0
                    elif iou >= negThr:
                        neg[nl + h, nw + v, z] = 0
                    v -= 1
                h += 1
            h = -1
            while nl + h >= 0:
                inter = anchors[nl + h, nw, z].intersection(gt).area
                iou = inter / (gtArea + anchorArea - inter)
                if iou < 0.1:
                    break
                if iou >= posThr:
                    pos[nl + h, nw, z] = 1
                    gi[nl + h, nw, z] = i
                    neg[nl + h, nw, z] = 0
                elif iou >= negThr:
                    neg[nl + h, nw, z] = 0
                v = 1
                while nw + v < anchors.shape[1]:
                    inter = anchors[nl + h, nw + v, z].intersection(gt).area
                    iou = inter / (gtArea + anchorArea - inter)
                    if iou < 0.1:
                        break
                    if iou >= posThr:
                        pos[nl + h, nw + v, z] = 1
                        gi[nl + h, nw + v, z] = i
                        neg[nl + h, nw + v, z] = 0
                    elif iou >= negThr:
                        neg[nl + h, nw + v, z] = 0
                    v += 1
                v = -1
                while nw + v >= 0:
                    inter = anchors[nl + h, nw + v, z].intersection(gt).area
                    iou = inter / (gtArea + anchorArea - inter)
                    if iou < 0.1:
                        break
                    if iou >= posThr:
                        pos[nl + h, nw + v, z] = 1
                        gi[nl + h, nw + v, z] = i
                        neg[nl + h, nw + v, z] = 0
                    elif iou >= negThr:
                        neg[nl + h, nw + v, z] = 0
                    v -= 1
                h -= 1
    return pos, neg, gi
    # pi = torch.where(pos)
    # ni = torch.where(neg)
    # return pi, ni, gi[pi]

def bboxCam2Lidar(camBoxes: Union[torch.Tensor, np.ndarray], c2v: Union[torch.Tensor, np.ndarray], inplace: bool = False):
    """
    Conver hwlxyzr format camera bboxes to xyzlwhr format lidar bboxes.
    @param inplace: Default False
    @param camBoxes: (N, 7) in 'hwlxyzr'
    @param c2v: (4, 4). Let v2c be calibration matrix 'Tr_velo_to_cam', which is in shape (3, 4),
    than c2v = np.linalg.inv(np.concatenate([v2c, [[0, 0, 0, 1]]], axis = 0))
    @return: (N, 7) in 'xyzlwhr'
    """
    if not inplace:
        t = torch.empty_like(camBoxes)
        t[...] = camBoxes
        camBoxes = t
    xyz = camBoxes[:, 3:6]
    xyz = torch.concat([xyz, torch.ones((camBoxes.shape[0], 1))], dim = 1).T
    xyz = c2v @ xyz
    xyz = xyz.T
    camBoxes[:, 3:6] = camBoxes[:, [2, 1, 0]]
    camBoxes[:, :3] = xyz[:, :3]
    camBoxes[:, 6] = -camBoxes[:, 6] - 0.5 * torch.pi
    camBoxes[:, 6] = torch.where(camBoxes[:, 6] < -torch.pi, camBoxes[:, 6] + 2 * torch.pi, camBoxes[:, 6])
    return camBoxes

def bboxLidar2Cam(lidarBoxes: torch.Tensor, v2c: torch.Tensor, inplace: bool = False):
    res: torch.Tensor
    if inplace:
        res = lidarBoxes
    else:
        res = torch.empty_like(lidarBoxes)
        res[...] = lidarBoxes
    xyz = res[:, :3]
    xyz = torch.concat([xyz, torch.ones((res.shape[0], 1))], dim = 1).T
    xyz = v2c @ xyz
    xyz = xyz.T
    res[:, :3] = res[:, [5, 4, 3]]
    res[:, 3:6] = xyz[:, :3]
    res[:, 6] = -res[:, 6] - 0.5 * torch.pi
    res[:, 6] = torch.where(res[:, 6] < -torch.pi, res[:, 6] + 2 * torch.pi, res[:, 6])
    return res

def decodeRegression(regmap: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    assert regmap.shape == anchors.shape
    d = torch.sqrt(anchors[..., [0]] ** 2 + anchors[..., [1]] ** 2)
    res = torch.empty(regmap.shape, device = regmap.device)
    res[..., :2] = regmap[..., :2] * d + anchors[..., :2]
    res[..., 2] = regmap[..., 2] * anchors[..., 5] + anchors[..., 2]
    res[..., 3:6] = torch.exp(regmap[..., 3:6]) * anchors[..., 3:6]
    res[..., 6] = torch.arcsin(regmap[..., 6]) + anchors[..., 6]
    return res