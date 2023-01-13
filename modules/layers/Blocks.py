from torch import nn
from torch.nn import functional as f
from modules import Config as cfg

class FCN(nn.Module):

    def __init__(self, cin, cout):
        super().__init__()
        self.fc = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm2d(cout, affine = cfg.bnaffine)

    def forward(self, x):
        # input shape = (batch, h, w, c)
        x = f.relu(self.fc(x))
        x = x.permute(0, 3, 1, 2)
        x = self.bn(x)
        # after BN: (batch, c, h, w)
        return x.permute(0, 2, 3, 1)

class CRB3d(nn.Module):

    def __init__(self, cin, cout, k, s, p):
        super().__init__()
        self.conv = nn.Conv3d(cin, cout, k, s, p)
        self.bn = nn.BatchNorm3d(cout, affine = cfg.bnaffine)

    def forward(self, x):
        x = f.relu(self.conv(x))
        return self.bn(x)

class CRB2d(nn.Module):

    def __init__(self, cin, cout, k, s, p):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, p)
        self.bn = nn.BatchNorm2d(cout, affine = cfg.bnaffine)

    def forward(self, x):
        x = f.relu(self.conv(x))
        return self.bn(x)

class DeCRB2d(nn.Module):

    def __init__(self, cin, cout, k, s, p):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(cin, cout, k, s, p)
        self.bn = nn.BatchNorm2d(cout, affine = cfg.bnaffine)

    def forward(self, x):
        x = f.relu(self.deconv(x))
        return self.bn(x)