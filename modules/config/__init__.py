from .Config import config
from .Parser import options, args
import os

dataroot = '/mnt/D/AIProject/mmdetection3d-master/data/kitti'
if len(args) > 0:
    dataroot = args[0]
veloroot = os.path.join(dataroot, 'training/velodyne_croped')
labelroot = os.path.join(dataroot, 'training/label_2')
calibroot = os.path.join(dataroot, 'training/calib')
imroot = os.path.join(dataroot, 'training/image_2')
trainInfoPath = os.path.join(dataroot, 'ImageSets/train.txt')
testInfoPath = os.path.join(dataroot, 'ImageSets/val.txt')

def __getattr__(name):
    return config[name]