from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn
from torch import nn
import torch
from torch.nn import functional as f
from modules import utils
from modules.layers import FCN, CRB2d

class ImageFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.__fasterRCNN = fasterrcnn_resnet50_fpn(True)
        self.__fasterRCNN.train(False)

    def forward(self, x):
        x = self.__fasterRCNN.transform(x)
        features = self.__fasterRCNN.backbone(x)
        features = (features['0'], features['1'], features['2'])
        return features

def featureMaping(voxels, features, calibs, imgsize):
    # input: voxel = batch * (N, T, C)
    # features = mapnum * (batch, C, H, W)
    # calibs = batch * {%calibration%}
    # assume that all images are of the same size
    # imgsize in (h, w)
    # return batch * (N, T, C * mapnum)

    if not isinstance(imgsize, torch.Tensor):
        imgsize = torch.Tensor(imgsize)
    features = features.transpose(0, 1) # transpose into (batch, mapnum, ...)

    regionSizes = []
    for feature in features:
        fhw = feature.shape[-2:]
        regionSize = fhw / imgsize
        regionSizes.append(regionSize)

    for i in range(len(features)):
        features[i] = f.pad(features[i], (0, 1, 0, 1))

    res = []
    for i, (v, c) in enumerate(zip(voxels, calibs)):
        origshape = v.shape[:-1]
        xyz = v[..., :3].reshape((-1, 3))
        proj = utils.lidar2Img(xyz, c)
        imageFeatures = []

        for feature, regionSize in zip(features, regionSizes):
            index = proj / regionSize
            index = index.floor()
            xi = 1 - proj[:, 0] + index[:, 0]   #
            yi = 1 - proj[:, 1] + index[:, 1]   #
            xi, yi = xi[None, :], yi[None, :]   # These are pre-calculated to
            xi_, yi_ = 1 - xi, 1 - yi           # reduce calculation times
            x, y = index[:, 0], index[:, 1]     #
            xplus1 = index[:, 0] + 1            #
            yplus1 = index[:, 1] + 1            #
            indexedFeature = feature[i, :, x, y] * xi * yi # shape = (C, S), S = proj.shape[0]
            indexedFeature = indexedFeature + feature[i, :, xplus1, y] * xi_ * yi
            indexedFeature = indexedFeature + feature[i, :, x, yplus1] * xi * yi_
            indexedFeature = indexedFeature + feature[i, :, xplus1, yplus1] * xi_ * yi_
            imageFeatures.append(indexedFeature)

        imageFeatures = torch.concat(imageFeatures, dim = 0).T
        imageFeatures = imageFeatures.reshape(origshape + (imageFeatures.shape[-1], ))
        res.append(imageFeatures)
    return res

class ImageFeatureFusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.fcn1 = FCN(768, 768)
        self.conv1 = CRB2d(768, 128, 1, 1, 0)
        self.fcn2 = FCN(128, 128)
        self.conv2 = CRB2d(128, 16, 1, 1, 0)
        self.fcn3 = FCN(16, 16)

    def forward(self, x):
        # input shape = (batch, N, T, C), C = 768
        x = self.fcn1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fcn2(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv2(x)
        x = x.permute(0, 2, 3, 1)
        x = self.fcn3(x)
        return x