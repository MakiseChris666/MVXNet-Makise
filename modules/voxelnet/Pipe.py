from torch import nn
import torch
# from modules.layers import FCRB, CRB2d, CRB3d, DeCRB2d
from modules.layers import FCBR, CBR2d, CBR3d, DeCBR2d
import modules.config as cfg

class VFE(nn.Module):

    def __init__(self, cin, cout, sampleNum):
        super().__init__()
        self.fcn = FCBR(cin, cout)
        self.sampleNum = sampleNum

    def forward(self, x):
        # input shape = (batch, N, 35, cin)
        x = self.fcn(x)
        # shape = (batch, N, 35, cout)
        s = torch.max(x, dim = 2, keepdim = True)[0].repeat(1, 1, self.sampleNum, 1)
        # concat on channels
        return torch.concat([x, s], dim = -1)

class SVFE(nn.Module):

    def __init__(self, sampleNum = 35):
        super().__init__()
        self.vfe1 = VFE(7, 16, sampleNum)
        self.vfe2 = VFE(32, 64, sampleNum)

    def forward(self, x):
        x = self.vfe1(x)
        return self.vfe2(x)

class CML(nn.Module):

    def __init__(self):
        super().__init__()
        self.downsample = self.downsampleLayers(cfg.voxelscale)
        self.conv1 = CBR3d(128, 128, 3, (2, 1, 1), (1, 1, 1))
        self.conv2 = CBR3d(128, 128, 3, 1, (0, 1, 1))
        self.conv3 = CBR3d(128, 128, 3, (2, 1, 1), 1)

    @staticmethod
    def downsampleLayers(downscale: int):
        if downscale <= 0:
            return lambda x: x
        else:
            layers = [CBR3d(128, 128, 2, 2, 0) for _ in range(downscale)]
            return nn.Sequential(*layers)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

try:
    from modules.layers import SparseCRB3d
    import spconv.pytorch as spconv
    class SparseCML(nn.Module):

        def __init__(self):
            super().__init__()
            self.downsample = self.downsampleLayers(cfg.voxelscale)
            self.subm1 = SparseCRB3d('subm', 128, 128, 3, 1, 1, bias = False)
            self.conv1 = SparseCRB3d('sparse', 128, 128, (3, 1, 1), (2, 1, 1), (1, 0, 0), bias = False)
            self.subm2 = SparseCRB3d('subm', 128, 128, 3, 1, 1, bias = False)
            self.subm3 = SparseCRB3d('subm', 128, 128, 3, 1, 1, bias = False)
            self.conv2 = SparseCRB3d('sparse', 128, 128, (3, 1, 1), (2, 1, 1), 0, bias = False)

        @staticmethod
        def downsampleLayers(downscale: int):
            if downscale <= 0:
                return lambda x: x
            else:
                layers = [SparseCRB3d('sparse', 128, 128, 2, 2, 0, bias = False) for _ in range(downscale)]
                return nn.Sequential(*layers)

        def forward(self, x):
            if not isinstance(x, spconv.SparseConvTensor):
                x = spconv.SparseConvTensor.from_dense(x)
            x = self.downsample(x)
            x = self.subm1(x)
            x = self.conv1(x)
            x = self.subm2(x)
            x = self.subm3(x)
            x = self.conv2(x)
            return x.dense()

except ImportError:
    pass

class RPN(nn.Module):

    def __init__(self):
        super().__init__()
        self.blk1 = nn.Sequential(
            CBR2d(256, 256, 3, 2, 1),
            *[CBR2d(256, 256, 3, 1, 1) for _ in range(2)]
        )
        self.blk2 = nn.Sequential(
            CBR2d(256, 256, 3, 2, 1),
            *[CBR2d(256, 256, 3, 1, 1) for _ in range(4)]
        )
        self.blk3 = nn.Sequential(
            CBR2d(256, 512, 3, 2, 1),
            *[CBR2d(512, 512, 3, 1, 1) for _ in range(4)]
        )
        self.deconv1 = DeCBR2d(256, 512, 3, 1, 1)
        self.deconv2 = DeCBR2d(256, 512, 2, 2, 0)
        self.deconv3 = DeCBR2d(512, 512, 4, 4, 0)
        self.cls = nn.Conv2d(1536, 2, 1, 1, 0)
        self.reg = nn.Conv2d(1536, 14, 1, 1, 0)
        self.dir = nn.Conv2d(1536, 4, 1, 1, 0)

    def forward(self, x):
        x1 = self.blk1(x)
        x2 = self.blk2(x1)
        x3 = self.blk3(x2)
        dx1 = self.deconv1(x1)
        dx2 = self.deconv2(x2)
        dx3 = self.deconv3(x3)
        x = torch.concat([dx1, dx2, dx3], dim = 1)
        return self.cls(x), self.reg(x), self.dir(x)
