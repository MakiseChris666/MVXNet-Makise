from torch import nn
import spconv.pytorch as spconv
import modules.config as cfg

class SparseCRB3d(nn.Module):

    def __init__(self, convtype: str, cin, cout, k, s, p, bias = True):
        super().__init__()
        if convtype == 'subm':
            conv = spconv.SubMConv3d
        elif convtype == 'sparse':
            conv = spconv.SparseConv3d
        else:
            raise NotImplemented
        self.seq = spconv.SparseSequential(
            conv(cin, cout, k, s, p, bias = bias),
            nn.ReLU(),
            nn.BatchNorm1d(cout, affine = cfg.bnaffine, track_running_stats = cfg.bntrack)
        )

    def forward(self, x):
        x = self.seq(x)
        return x