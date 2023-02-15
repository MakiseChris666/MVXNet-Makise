from torch import nn
import torch
from torch.nn import SmoothL1Loss, CrossEntropyLoss
import modules.config as cfg
import torch.nn.functional as f
from torchvision.ops.focal_loss import sigmoid_focal_loss

class VoxelLoss(nn.Module):

    def __init__(self, beta = 0.4, eps = cfg.eps, lossWeights = (1.0, 2.0, 0.2)):
        super().__init__()
        self.eps = eps
        self.lossWeights = lossWeights
        self.smoothl1 = SmoothL1Loss(beta = beta, reduction = 'sum')
        self.angleloss = lambda x: torch.where(x < beta, 0.5 * x ** 2 / beta, x - 0.5 * beta).sum()
        self.entropy = CrossEntropyLoss()

    def forward(self, pi, ni, gi, gts, score, reg, dir, anchors, anchorsPerLoc):

        dir = dir.reshape((cfg.voxelshape[0] // 2, cfg.voxelshape[1] // 2, 2, 2))
        dir = f.softmax(dir, dim = -1)
        clsTar = torch.zeros_like(score, dtype = cfg.dtype, device = cfg.device)
        clsTar[pi] = 1

        clsLoss = sigmoid_focal_loss(score, clsTar)
        valid = torch.ones_like(score, dtype = cfg.dtype, device = cfg.device)
        valid[ni] = 0
        valid[pi] = 1
        clsLoss = clsLoss * valid
        clsLoss = clsLoss.sum() / max(1, pi[0].shape[0])

        # score = torch.sigmoid(score)
        # posLoss = -(torch.log(score[pi] + self.eps) * (1 - score[pi]) ** 2).sum()
        # negLoss = -torch.log(1 - score + self.eps) * score ** 2
        # sizeSum = negLoss.shape[0] * negLoss.shape[1] * negLoss.shape[2]
        # negLoss = negLoss.sum() - negLoss[ni].sum()
        # posLoss = posLoss / (pi[0].shape[0] + self.eps)
        # negLoss = negLoss / (sizeSum - ni[0].shape[0] + self.eps)
        # clsLoss = 0.25 * posLoss + 0.75 * negLoss

        if len(pi[0]) == 0:
            return clsLoss * self.lossWeights[0], None, None

        alignedGTs = gts[gi]
        anchors = anchors.reshape((anchors.shape[0], anchors.shape[1], anchorsPerLoc, 7))
        alignedAnchors = anchors[pi]
        d = torch.sqrt(alignedAnchors[:, 3] ** 2 + alignedAnchors[:, 4] ** 2)[:, None].type(cfg.dtype)
        targets = torch.empty((alignedGTs.shape[0], 6), dtype = cfg.dtype, device = cfg.device)
        targets[:, [0, 1]] = (alignedGTs[:, [0, 1]] - alignedAnchors[:, [0, 1]]) / d
        targets[:, 2] = (alignedGTs[:, 2] - alignedAnchors[:, 2]) / alignedAnchors[:, 5]
        targets[:, 3:6] = torch.log(alignedGTs[:, 3:6] / alignedAnchors[:, 3:6])
        # targets[:, 6] = torch.sin(alignedGTs[:, 6] - alignedAnchors[:, 6])
        dirpos = alignedGTs[:, 6] > 0

        reg = reg.reshape((reg.shape[0], reg.shape[1], anchorsPerLoc, 7))[pi]
        regLoss = (self.smoothl1(reg[:, :6], targets) +
                  self.angleloss(torch.sin(reg[:, 6] - alignedGTs[:, 6]))) / pi[0].shape[0]

        dir = dir[pi]
        dirTar = torch.zeros_like(dir, dtype = cfg.dtype, device = cfg.device)
        dirTar[:, 1] = 1
        dirTar[dirpos] = 1 - dirTar[dirpos]
        dirLoss = self.entropy(dir, dirTar)

        return clsLoss * self.lossWeights[0], regLoss * self.lossWeights[1], dirLoss * self.lossWeights[2]
