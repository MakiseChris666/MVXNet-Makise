import numpy as np
import modules.Config as cfg
from numba import njit
from typing import Tuple, List
from modules.Calc import bbox3d2bev
from modules.Extension import cpp
from modules.utils import bboxIntersection
import torch
from torchvision.ops.boxes import box_area
import cv2 as cv

@njit
def check(pcd: np.ndarray, velorange: List[float], gridshape: Tuple[int, int] = (704, 800)):
    # velorange = np.array(velorange)
    low = np.array(velorange[:2])
    gridsize = np.array([(velorange[3] - velorange[0]) / gridshape[0], (velorange[4] - velorange[1]) / gridshape[1]])
    loc = ((pcd[:, :2] - low) / gridsize).astype(np.int32)
    zmax = np.empty(gridshape)
    zmax[...] = velorange[2] - 1
    for l, p in zip(loc, pcd):
        zmax[l[0], l[1]] = zmax[l[0], l[1]] if zmax[l[0], l[1]] > p[2] else p[2]
    return zmax

IOF_THRS = (0.1, 0.3, 0.5)
fail = {0.1: 0, 0.3: 0, 0.5: 0}

def locate(scenepcd, scenebevs, scenebbox2ds, scenebbox3ds, gts, iterlim = 30):
    zmax = check(scenepcd, cfg.velorange)
    chosen = np.random.choice(gts, iterlim, replace = False)
    sarea = box_area(scenebbox2ds)
    curThr = np.random.choice(IOF_THRS)
    lowx, lowy, lowz = cfg.velorange[0], cfg.velorange[1], cfg.velorange[2]
    for gt in chosen:
        gtbbox3d = gt['bbox3d']
        gtbbox2d = gt['bbox2d']

        gtx, gty = gtbbox3d[:2]
        gridx, gridy = (gtx - lowx) / 0.1, (gty - lowy) / 0.1
        gridx, gridy = int(gridx), int(gridy)
        zground = zmax[gridx, gridy]
        if zground > gtbbox3d[2] + 0.1:
            continue

        gtbev = bbox3d2bev(gtbbox3d)
        if scenebevs.shape[0] == 0:
            gt['bev'] = gtbev
            return gt

        inter = bboxIntersection(scenebbox2ds, gtbbox2d[None, :]).squeeze(dim = 1)
        iof = inter / sarea
        if torch.max(iof) > curThr:
            continue

        bevious = cpp.bboxOverlap(gtbev[None, :], scenebevs)
        if bevious.max() > 0.05:
            continue
        gt['bev'] = gtbev
        return gt
    fail[curThr] += 1
    return None

def augment(pcd, img, scenebbox2ds, scenebbox3ds, scenebevs, gts, lim):
    if scenebbox2ds is None:
        scenebbox2ds = torch.empty((0, 4))
        scenebbox3ds = torch.empty((0, 7))
        scenebevs = torch.empty((0, 4, 2))
    if lim < scenebbox3ds.shape[0]:
        return [], [], img, scenebbox3ds, scenebevs
    img = img.copy()
    resvelo = []
    rescalib = []
    for i in range(lim - scenebbox3ds.shape[0]):
        gt = locate(pcd, scenebevs, scenebbox2ds, scenebbox3ds, gts)
        if gt is None:
            continue
        resvelo.append(gt['velo'])
        rescalib.append(gt['calib'])
        scenebevs = torch.concat([scenebevs, gt['bev'][None, ...]], dim = 0)
        scenebbox2ds = torch.concat([scenebbox2ds, gt['bbox2d'][None, ...]], dim = 0)
        scenebbox3ds = torch.concat([scenebbox3ds, gt['bbox3d'][None, ...]], dim = 0)
        maskbbox = gt['maskbbox']
        mask = gt['mask']
        gtimg = gt['image']
        gtimg = cv.bitwise_and(gtimg, gtimg, mask = mask)
        imgroi = img[maskbbox[1]:maskbbox[3] + 1, maskbbox[0]:maskbbox[2] + 1]
        maskinv = 1 - mask
        imgroi = cv.bitwise_and(imgroi, imgroi, mask = maskinv)
        imgroi = cv.add(imgroi, gtimg)
        img[maskbbox[1]:maskbbox[3] + 1, maskbbox[0]:maskbbox[2] + 1] = imgroi
    return resvelo, rescalib, img, scenebbox3ds, scenebevs

def augmentTargetClasses(pcd, img, bbox2ds, bbox3ds, bevs, gtwithinfo, targets, lims):
    """
    Augment data
    @param pcd: Point cloud
    @param img: Camera image
    @param bbox2ds: 2d bounding boxes in xyxy
    @param bbox3ds: 3d bounding boxes in xyzlwhr
    @param bevs: bevs in corner
    @param gtwithinfo: 'dict' type, keys are class names, values are lists of gts
    @param targets: Target classes
    @param lims: Number of gt samples to be in the scene
    @return: A list of gt pcds to be augmented; a list of corresponding calib matrices;
            the image after augmentation; "all" 3d bboxes in dict[class name, bboxes];
            "all" bevs corresponding to 3d bboxes
    """
    augvelos, augcalibs, augbbox3ds, augbevs = [], [], {}, {}
    for c, l in zip(targets, lims):
        augvelo, augcalib, img, augbbox3d, augbev = augment(pcd, img, bbox2ds, bbox3ds, bevs, gtwithinfo[c], l)
        augvelos.extend(augvelo)
        augcalibs.extend(augcalib)
        augbbox3ds[c] = augbbox3d
        augbevs[c] = augbev
    return augvelos, augcalibs, img, augbbox3ds, augbevs