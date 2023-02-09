from torchvision.models.detection.faster_rcnn import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torch import nn
import torch
from torch.nn import functional as f
from modules.layers import FCRB, CRB2d
import modules.config as cfg

_fasterRCNN = fasterrcnn_resnet50_fpn_v2(weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)

class ImageFeatureExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.transform = _fasterRCNN.transform
        self.backbone = _fasterRCNN.backbone

    def forward(self, x):
        x, _ = self.transform(x)
        features = self.backbone(x.tensors)
        features = [features['0'], features['1'], features['2']]
        return features

def featureMaping(voxels, features, calibs, imsize):
    # input: voxel = batch * (N, T, C)
    # features = mapnum * (batch, C, H, W)
    # calibs = batch * {%calibration%}
    # assume that all images are of the same size
    # imsize in (h, w)
    # return batch * (N, T, C * mapnum)
    """
    Map image features to corresponding voxels \n
    BTW, this function will set zero voxel points(i.e. x=y=z=0) to all zero.
    This functionality is implemented here for convenience
    @param voxels: batch * (N, T, C)
    @param features: mapnum * (batch, C1, H, W)
    @param calibs: batch * calib_dict
    @param imsize: (h, w), note that it show be the same type as voxels and features
    @return: batch * (N, T, C * mapnum)
    """

    regionSizes = []
    for feature in features:
        fhw = torch.Tensor([*feature.shape[-2:]]).to(imsize.device)
        regionSize = imsize / fhw
        regionSizes.append(regionSize)

    for i in range(len(features)):
        features[i] = f.pad(features[i], (0, 1, 0, 1))

    res = []
    for i, (v, c) in enumerate(zip(voxels, calibs)):
        origshape = v.shape[:-1]
        xyz = v[..., :3].reshape((-1, 3))
        zero = torch.all(xyz == 0, dim = 1)
        proj = v[..., -2:].reshape((-1, 2))
        proj[zero] = 0
        imageFeatures = []
        zero = zero.reshape(origshape)
        v[zero] = 0

        for feature, regionSize in zip(features, regionSizes):
            indexNonaligned = proj / regionSize - cfg.eps
            index = indexNonaligned.long()
            xi = indexNonaligned[:, 0] - index[:, 0]        #
            yi = indexNonaligned[:, 1] - index[:, 1]        #
            xi, yi = xi[None, :], yi[None, :]               # These are pre-calculated to
            xi_, yi_ = 1 - xi, 1 - yi                       # reduce calculation times
            x, y = index[:, 0], index[:, 1]                 #
            xplus1 = index[:, 0] + 1                        #
            yplus1 = index[:, 1] + 1                        #
            assert torch.max(xplus1) < feature.shape[-2] and torch.max(yplus1) < feature.shape[-1]
            indexedFeature = feature[i, :, x, y] * xi * yi # shape = (C, S), S = proj.shape[0]
            indexedFeature = indexedFeature + feature[i, :, xplus1, y] * xi_ * yi
            indexedFeature = indexedFeature + feature[i, :, x, yplus1] * xi * yi_
            indexedFeature = indexedFeature + feature[i, :, xplus1, yplus1] * xi_ * yi_
            imageFeatures.append(indexedFeature)

        imageFeatures = torch.concat(imageFeatures, dim = 0).T
        imageFeatures = imageFeatures.reshape(origshape + (imageFeatures.shape[-1], ))
        imageFeatures[zero] = 0
        res.append(imageFeatures)
    return res

class ImageFeatureFusion(nn.Module):

    def __init__(self):
        super().__init__()
        self.fcn1 = FCRB(768, 768)
        self.fcn2 = FCRB(768, 128)
        self.fcn3 = FCRB(128, 128)
        self.fcn4 = FCRB(128, 16)
        self.fcn5 = FCRB(16, 16)

    def forward(self, x):
        # input shape = (batch, N, T, C), C = 768
        x = self.fcn1(x)
        x = self.fcn2(x)
        x = self.fcn3(x)
        x = self.fcn4(x)
        x = self.fcn5(x)
        return x