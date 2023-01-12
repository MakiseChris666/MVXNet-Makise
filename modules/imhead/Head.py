from Pipe import *
from torch import nn

class ImageHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.extractor = ImageFeatureExtractor()
        self.fusion = ImageFeatureFusion()

    def forward(self, x, voxels):
        x = self.extractor(x)
