import json
import cv2 as cv
import numpy as np
# from open3d.cpu.pybind.visualization import draw_geometries
from pycocotools import mask as mask_utils
import os
from torchvision.ops import box_iou
from open3d.cpu.pybind.utility import Vector3dVector
from open3d.cpu.pybind.geometry import PointCloud, OrientedBoundingBox
import sys
import pandas as pd
import torch
from modules.Calc import bboxCam2Lidar, bbox3d2corner
import pickle as pkl
from modules import Config as cfg

def polys_to_mask(polygons, height, width):
    """
    Function from https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset
    @param polygons:
    @param height:
    @param width:
    @return:
    """
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    mask = mask_utils.decode(rle)
    return mask

def make_json_dict(imgs, anns):
    """
    Funcion from https://github.com/qqlu/Amodal-Instance-Segmentation-through-KINS-Dataset
    @param imgs:
    @param anns:
    @return:
    """
    imgs_dict = {}
    anns_dict = {}
    for ann in anns:
        image_id = ann["image_id"]
        if not image_id in anns_dict:
            anns_dict[image_id] = []
            anns_dict[image_id].append(ann)
        else:
            anns_dict[image_id].append(ann)

    for img in imgs:
        image_id = img['id']
        imgs_dict[image_id] = img['file_name']

    return imgs_dict, anns_dict

with open('./seglabel/update_train_2020.json', 'r') as f:
    segInfo = json.load(f)

imgsInfo = segInfo['images']
annsInfo = segInfo['annotations']

dataroot = '../mmdetection3d-master/data/kitti'
if len(sys.argv) > 1 and sys.argv[1] != '-':
    dataroot = sys.argv[1]
veloroot = os.path.join(dataroot, 'training/velodyne_croped')
labelroot = os.path.join(dataroot, 'training/label_2')
calibroot = os.path.join(dataroot, 'training/calib')
imroot = os.path.join(dataroot, 'training/image_2')
gtroot = os.path.join(dataroot, 'training/gtdatabase')
trainInfoPath = os.path.join(dataroot, 'ImageSets/train.txt')
with open(trainInfoPath, 'r') as f:
    trainSet = set(f.read().splitlines())
if not os.path.exists(gtroot):
    os.mkdir(gtroot)

velorange = cfg.velorange
rangeMin = torch.Tensor(velorange[:3])[None, :]
rangeMax = torch.Tensor(velorange[3:])[None, :]
imsize = cfg.imsize

imgsDict, annsDict = make_json_dict(imgsInfo, annsInfo)
tarClasses = ['Car', 'Pedestrian', 'Cyclist']
clsToId = {'Car': 4, 'Pedestrian': 2, 'Cyclist': 1}
idToCls = {1: 'Cyclist', 2: 'Pedestrian', 4: 'Car'}

insroot = {}
for c in tarClasses:
    insroot[clsToId[c]] = os.path.join(gtroot, c)
    if not os.path.exists(insroot[clsToId[c]]):
        os.mkdir(insroot[clsToId[c]])
gtinforoot = os.path.join(gtroot, 'gtinfo.pkl')
gtinfo = {
    'Car': [],
    'Cyclist': [],
    'Pedestrian': []
}

cnt = {1:0, 2:0, 4:0}
p = 0

print('This script will create a database of ground truth samples for data augmentation.')

for id in annsDict.keys():
    imgName = imgsDict[id]
    s = imgName[:6]
    if s not in trainSet:
        continue
    p += 1
    print(f'\rCreating ground truth database: {p}/3707', end = '')

    imgPath = os.path.join(imroot, imgName)
    img = cv.imread(imgPath)
    h, w = img.shape[:2]
    img = img[:imsize[0], :imsize[1]]

    path = os.path.join(veloroot, s + '.bin')
    velo = np.fromfile(path, dtype = 'float32').reshape((-1, 4))
    pcd = PointCloud(Vector3dVector(velo[:, :3]))
    pcd.colors = Vector3dVector(np.tile(velo[:, [3]], (1, 3)))

    path = os.path.join(labelroot, s + '.txt')
    labels = pd.read_csv(path, sep = ' ', index_col = 0, header = None)
    labels = labels[labels.index.isin(tarClasses)]
    kittiBoxes = {}
    occlude = {}

    path = os.path.join(calibroot, s + '.txt')
    calib = {}
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        l = lines[5].split(' ')
        v2c = np.array(l[1:]).astype('float32').reshape((3, 4))
        v2c = np.concatenate([v2c, [[0, 0, 0, 1]]], axis = 0)
        calib[l[0][:-1]] = v2c
        l = lines[2].split(' ')
        p2 = np.array(l[1:]).astype('float32').reshape((3, 4))
        p2 = np.concatenate([p2, [[0, 0, 0, 1]]], axis = 0)
        calib[l[0][:-1]] = p2
        l = lines[4].split(' ')
        r0 = np.zeros((4, 4))
        r0[:3, :3] = np.array(l[1:]).astype('float32').reshape((3, 3))
        r0[3, 3] = 1
        calib[l[0][:-1]] = r0

    c2v = np.linalg.inv(calib['Tr_velo_to_cam'])
    c2v = torch.Tensor(c2v)

    for c in tarClasses:
        l = labels[labels.index == c].to_numpy()
        if len(l) == 0:
            kittiBoxes[clsToId[c]] = None

        l = torch.Tensor(l)
        l[:, 7:] = bboxCam2Lidar(l[:, 7:], c2v, True)
        inRange = torch.all(l[:, 7:10].__lt__(rangeMax), dim = 1) & \
                  torch.all(l[:, 7:10].__ge__(rangeMin), dim = 1)
        l = l[inRange]
        kittiBoxes[clsToId[c]] = l[:, 3:] if len(l) > 0 else None
        occlude[clsToId[c]] = l[:, 1]

    masks = {1: [], 2: [], 4: []}
    maskBoxes = {1: [], 2: [], 4: []}

    for ann in annsDict[id]:

        cls = ann['category_id']
        if cls not in masks:
            continue
        mask = polys_to_mask(ann['i_segm'], h, w) # ndarray
        mask = mask[:imsize[0], :imsize[1]]
        bbox = ann['a_bbox']
        bbox[2] += bbox[0]
        bbox[3] += bbox[1]
        masks[cls].append(mask)
        maskBoxes[cls].append(bbox)

    for k in kittiBoxes.keys():
        kittiBox = kittiBoxes[k] # original labels
        if kittiBox is None:
            continue

        maskBox = torch.Tensor(maskBoxes[k]) # 2d bboxes for masks
        if len(maskBox) == 0:
            continue
        ious = box_iou(kittiBox[:, :4], maskBox)

        maxious, indices = torch.max(ious, dim = 1)
        f = maxious >= 0.65
        indices = indices[f]
        kittiBox = kittiBox[f]
        maskBox = maskBox[indices]
        if len(maskBox) == 0:
            continue

        bbox3ds = kittiBox[:, 4:]
        corners = bbox3d2corner(bbox3ds).numpy()
        bbox2ds = kittiBox[:, :4]

        curmasks = masks[k]
        maskBox = maskBox.int()
        occ = occlude[k]
        occ = occ[f]

        for i, c, m, o, bbox3d, bbox2d in zip(indices, corners, maskBox, occ, bbox3ds, bbox2ds):
            mask = curmasks[i]
            imgroi = img[m[1]:m[3] + 1, m[0]:m[2] + 1]
            maskroi = mask[m[1]:m[3] + 1, m[0]:m[2] + 1]
            imgroi = imgroi * maskroi[..., None]
            if imgroi.shape[0] == 0 or imgroi.shape[1] == 0:
                continue
            # cv.imshow('', imgroi)

            cropBox = OrientedBoundingBox().create_from_points(Vector3dVector(c))
            gtpcd = pcd.crop(cropBox)
            # draw_geometries([gtpcd])
            gtvelo = np.asarray(gtpcd.points).astype('float32')
            r = np.asarray(gtpcd.colors)[:, [0]].astype('float32')
            gtvelo = np.concatenate([gtvelo, r], axis = 1)

            veloname = 'velo_%06d.bin' % cnt[k]
            imgname = 'img_%06d.png' % cnt[k]
            maskname = 'mask_%06d.npy' % cnt[k]
            gtinfo[idToCls[k]].append({
                'velo': veloname,
                'image': imgname,
                'mask': maskname,
                'occlude': o,
                'maskbbox': m,
                'bbox2d': bbox2d,
                'bbox3d': bbox3d,
                'id': s
            })

            gtvelo.tofile(os.path.join(insroot[k], veloname))
            cv.imwrite(os.path.join(insroot[k], imgname), imgroi)
            np.save(os.path.join(insroot[k], maskname), maskroi)
            cnt[k] += 1
            # cv.destroyAllWindows()

with open(gtinforoot, 'wb') as f:
    pkl.dump(gtinfo, f)