import os
import os.path as osp
import numpy as np
import random
import collections
import torch
import torchvision
import cv2
from torch.utils import data
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize
import SimpleITK as sitk
import math

import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
import glob
import imgaug.augmenters as iaa


import matplotlib.pyplot as plt
from skimage.transform import rescale, resize
import glob
from torch.utils.data import DataLoader, random_split
import scipy.ndimage
import cv2
import PIL
import sys
import pandas as pd


class MOTSDataSet(data.Dataset):
    def __init__(self, supervise_root, list_path, max_iters=None, crop_size=(64, 192, 192), mean=(128, 128, 128), scale=True,
                 mirror=True, ignore_label=255, edge_weight = 1):
        self.supervise_root = supervise_root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.edge_weight = edge_weight

        self.image_mask_aug = iaa.Sequential([
            iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
            iaa.Affine(rotate=(-180, 180)),
            iaa.Affine(shear=(-16, 16)),
            iaa.Fliplr(0.5),
            iaa.ScaleX((0.75, 1.5)),
            iaa.ScaleY((0.75, 1.5))
        ])

        self.image_aug_color = iaa.Sequential([
            # iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
            iaa.GammaContrast((0, 2.0)),
            iaa.Add((-0.1, 0.1), per_channel=0.5),
            #iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)), # new
            #iaa.AddToHueAndSaturation((-0.1, 0.1)),
            #iaa.GaussianBlur(sigma=(0, 1.0)), # new
            #iaa.AdditiveGaussianNoise(scale=(0, 0.1)), # new
        ])

        self.image_aug_noise = iaa.Sequential([
            # iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True),
            #iaa.GammaContrast((0.5, 2.0)),
            #iaa.Add((-0.1, 0.1), per_channel=0.5),
            iaa.CoarseDropout((0.0, 0.05), size_percent=(0.00, 0.25)),  # new
            # iaa.AddToHueAndSaturation((-0.1, 0.1)),
            iaa.GaussianBlur(sigma=(0, 1.0)),  # new
            iaa.AdditiveGaussianNoise(scale=(0, 0.1)),  # new
        ])

        self.image_aug_resolution = iaa.AverageBlur(k=(2, 8))

        self.image_aug_256 = iaa.Sequential([
            iaa.MultiplyHueAndSaturation((-10, 10), per_channel=0.5)
        ])


        'supervise'
        self.df_supervise = pd.read_csv(self.supervise_root)
        self.df_supervise.sample(frac=1)

        self.now_len = len(self.df_supervise)

        print('{} images are loaded!'.format(self.now_len))

    def __len__(self):
        return self.now_len# len(self.files)

    def __getitem__(self, index):
        'supervised'
        datafiles = self.df_supervise.iloc[index]

        image = plt.imread(datafiles["image_path"])
        label = plt.imread(datafiles["label_path"])

        name = datafiles["name"]
        task_id = datafiles["task_id"]
        scale_id = datafiles["scale_id"]

        # data augmentation
        image = image[:,:,:3]
        label = label[:,:,:3]

        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        seed = np.random.rand(4)

        if seed[0] > 0.5:
            image, label = self.image_mask_aug(images=image, heatmaps=label)

        if seed[1] > 0.5:
            image = self.image_aug_color(images=image)

        if seed[2] > 0.5:
            image = self.image_aug_noise(images=image)

        label[label >= 0.5] = 1.
        label[label < 0.5] = 0.

        image = image[0].transpose((2, 0, 1))  # Channel x H x W
        label = label[0,:,:,0]

        image = image.astype(np.float32)
        label = label.astype(np.uint8)

        if (self.edge_weight):
            weight = scipy.ndimage.morphology.binary_dilation(label == 1, iterations=2) & ~ label
        else:  # otherwise the edge weight is all ones and thus has no affect
            weight = np.ones(label.shape, dtype=label.dtype)

        label = label.astype(np.float32)

        return image.copy(), label.copy(), weight.copy(), name, task_id, scale_id


class MOTSValDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(256, 256), mean=(128, 128, 128), scale=False,
                 mirror=False, ignore_label=255, edge_weight = 1):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.edge_weight = edge_weight

        self.df = pd.read_csv(self.root)
        self.df.sample(frac=1)

        print('{} images are loaded!'.format(len(self.df)))

    def __len__(self):
        return len(self.df)


    def __getitem__(self, index):
        datafiles = self.df.iloc[index]
        # read png file
        image = plt.imread(datafiles["image_path"])
        label = plt.imread(datafiles["label_path"])

        name = datafiles["name"]
        task_id = datafiles["task_id"]
        scale_id = datafiles["scale_id"]

        # data augmentation
        image = image[:,:,:3]
        label = label[:,:,:3]

        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        label[label >= 0.5] = 1.
        label[label < 0.5] = 0.

        # image = image.transpose((3, 1, 2, 0))  # Channel x H x W
        # label = label[:,:,:,0].transpose((1, 2, 0))

        image = image[0].transpose((2, 0, 1))  # Channel x H x W
        label = label[0,:,:,0]

        image = image.astype(np.float32)
        label = label.astype(np.float32)
        weight = np.ones(label.shape, dtype=label.dtype)

        return image.copy(), label.copy(), weight.copy(),  name, task_id, scale_id

def my_collate(batch):
    image, label, weight, name, task_id, scale_id= zip(*batch)
    image = np.stack(image, 0)
    label = np.stack(label, 0)
    name = np.stack(name, 0)
    weight = np.stack(weight, 0)
    task_id = np.stack(task_id, 0)
    scale_id = np.stack(scale_id, 0)
    data_dict = {'image': image, 'label': label, 'weight': weight, 'name': name, 'task_id': task_id, 'scale_id': scale_id}
    #tr_transforms = get_train_transform()
    #data_dict = tr_transforms(**data_dict)
    return data_dict

if __name__ == '__main__':

    trainset_dir = '/Data2/KI_data_trainingset_patch/data_list.csv'
    semi_dir = '/Data2/KI_Semi_patch/data_list.csv'
    train_list = '/Data2/KI_data_trainingset_patch/data_list.csv'
    itrs_each_epoch = 250
    batch_size = 1
    input_size = (256,256)
    random_scale = False
    random_mirror = False

    save_img = '/media/dengr/Data2/KI_data_test_patches'
    save_mask = '/media/dengr/Data2/KI_data_test_patches'

    img_scale = 0.5

    trainloader = DataLoader(
        MOTSDataSet(trainset_dir, semi_dir, train_list, max_iters=itrs_each_epoch * batch_size,
                    crop_size=input_size, scale=random_scale, mirror=random_mirror),batch_size = 4, shuffle = False, num_workers =0)

    for i in range(0,10):
        for iter, batch in enumerate(trainloader):
            print('aaa')
        # imgs = torch.from_numpy(batch['image']).cuda()
        # lbls = torch.from_numpy(batch['label']).cuda()
        # volumeName = batch['name']
        # t_ids = torch.from_numpy(batch['task_id']).cuda()
        # s_ids = torch.from_numpy(batch['scale_id']).cuda()
