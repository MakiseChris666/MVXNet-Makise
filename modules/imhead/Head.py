from .Pipe import *
from torch import nn

class ImageHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.extractor = ImageFeatureExtractor()
        self.fusion = ImageFeatureFusion()

    def forward(self, x, voxels, calibs, imsize):
        # input: voxels = batch * (...), calibs = batch * {...}
        x = self.extractor(x)
        imfeatures = featureMaping(voxels, x, calibs, imsize)
        # here imf is a list, actually of length 1
        imfeatures = imfeatures[0][None, ...] # turn it into a whole tensor
        imfeatures = self.fusion(imfeatures)
        # returns (batch, N, T, 16), batch now equals 1
        return imfeatures