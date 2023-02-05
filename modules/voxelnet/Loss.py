from torch import nn
import torch
from torch.nn import SmoothL1Loss
import modules.config as cfg
import torch.nn.functional as f

class VoxelLoss(nn.Module):

    def __init__(self, a = 1.5, b = 1, eps = cfg.eps):
        super().__init__()
        self.a = a
        self.b = b
        self.eps = eps
        self.smoothl1 = SmoothL1Loss()

    def forward(self, pi, ni, gi, gts, score, reg, anchors, anchorsPerLoc):

        score = score.reshape((176, 200, 2, 2))
        score = f.softmax(score, dim = -1)
        pos = score[..., 0]
        neg = score[..., 1]

        if pi is None:
            clsLoss = -torch.log(neg + self.eps).mean()
            return clsLoss, None

        posLoss = -torch.log(pos[pi] + self.eps).sum()
        negLoss = -torch.log(neg + self.eps)
        sizeSum = negLoss.shape[0] * negLoss.shape[1] * negLoss.shape[2]
        negLoss = negLoss.sum() - negLoss[ni].sum()
        posLoss = posLoss / (pi[0].shape[0] + self.eps)
        negLoss = negLoss / (sizeSum - ni[0].shape[0] + self.eps)
        clsLoss = self.a * posLoss + self.b * negLoss

        if len(pi[0]) == 0:
            return clsLoss, None

        alignedGTs = gts[gi]
        anchors = anchors.reshape((anchors.shape[0], anchors.shape[1], anchorsPerLoc, 7))
        alignedAnchors = anchors[pi]
        d = torch.sqrt(alignedAnchors[:, 3] ** 2 + alignedAnchors[:, 4] ** 2)[:, None].type(cfg.dtype)
        targets = torch.empty_like(alignedGTs, dtype = cfg.dtype, device = cfg.device)
        targets[:, [0, 1]] = (alignedGTs[:, [0, 1]] - alignedAnchors[:, [0, 1]]) / d
        targets[:, 2] = (alignedGTs[:, 2] - alignedAnchors[:, 2]) / alignedAnchors[:, 5]
        targets[:, 3:6] = torch.log(alignedGTs[:, 3:6] / alignedAnchors[:, 3:6])
        targets[:, 6] = alignedGTs[:, 6] - alignedAnchors[:, 6]

        reg = reg.reshape((reg.shape[0], reg.shape[1], anchorsPerLoc, 7))[pi]
        regLoss = self.smoothl1(reg, targets)

        return clsLoss, regLoss
