from torch import nn
from modules.voxelnet import VoxelNet
from modules.imhead import ImageHead
import torch

__all__ = ['MVXNet']

def initWeights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()

class MVXNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.head = ImageHead()
        self.backbone = VoxelNet()
        self.backbone.apply(initWeights)

    def forward(self, voxels, imgs, idx, calibs, imsize):
        # calibs = batch * calib_dict
        # actually it supports batchsize = 1 currently, so calibs = [calib_dict]
        # so voxels = (1, ...), it behaves the same as a single-element tuple so that's ok
        imfeatures = self.head(imgs, voxels, calibs, imsize) # (batch, N, T, 16)
        voxels = torch.concat([voxels, imfeatures], dim = -1)
        x = self.backbone(voxels, idx)
        return x