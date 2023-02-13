import torch
from modules.Extension import cpp
import numpy as np
from modules.Calc import bbox3d2bev
from numba import njit

@njit
def _nms(score: np.ndarray, ious: np.ndarray, iouThr: float = 0.01):
    out = ious > iouThr
    cur = np.zeros_like(score, dtype = 'bool')
    res = []
    for i in range(len(score)):
        if cur[i]:
            continue
        res.append(i)
        cur = cur | out[i]
    return np.array(res)

def nmsbev(score: torch.Tensor, bbox3ds: torch.Tensor):
    score, indices = torch.sort(score, descending = True)
    bbox3ds = bbox3ds[indices]
    bbox3ds = bbox3d2bev(bbox3ds).numpy()
    ious = cpp.bboxOverlap(bbox3ds, bbox3ds)
    reserved = _nms(score.numpy(), ious)
    return indices[reserved]