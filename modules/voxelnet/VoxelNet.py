import torch
from torch import nn
from torch.autograd import Variable
from modules import Config as cfg
from modules.voxelnet import Pipe

class VoxelNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.svfe = Pipe.SVFE(cfg.samplenum)
        self.fcn = Pipe.FCN(128, 128)
        self.cml = Pipe.CML()
        self.rpn = Pipe.RPN()

    @staticmethod
    def reindex(x, idx):
        # input shape: x = (batch * N, 128), idx = (batch * N, 1 + 3)
        res = Variable(torch.empty((1, 128, cfg.voxelshape[2], cfg.voxelshape[0]
                                    , cfg.voxelshape[1]), dtype = cfg.dtype, device = cfg.device))
        res[idx[:, 0], :, idx[:, 3], idx[:, 1], idx[:, 2]] = x
        return res

    def forward(self, x, idx):
        # x shape = (batch, N, 35, 7), idx shape = (batch * N, 1 + 3), idx[:, 0] is the batch No.
        # idx: corresponding indices to voxels
        x = self.svfe(x)
        x = self.fcn(x)
        # after SVFE&FCN: shape = (batch, N, 35, 128)
        x = torch.max(x, dim = 2)[0]
        # after elementwise max: shape = (batch, N, 1, 128)
        x = torch.squeeze(x, dim = 2).reshape((-1, 128))
        # shape = (batch * N, 128)
        x = self.reindex(x, idx)
        x = self.cml(x)
        x = x.reshape((1, -1, cfg.voxelshape[0], cfg.voxelshape[1]))
        score, reg = self.rpn(x)
        return score, reg