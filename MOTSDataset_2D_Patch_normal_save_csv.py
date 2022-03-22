import os
import os.path as osp
import numpy as np
import random
import collections

import pandas as pd
import torch
import torchvision
import cv2
from torch.utils import data
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize
import SimpleITK as sitk
import math
# from batchgenerators.transforms import Compose
# from batchgenerators.transforms.spatial_transforms import SpatialTransform, MirrorTransform
# from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform, \
#     BrightnessTransform, ContrastAugmentationTransform
# from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
# from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
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


class MOTSDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(64, 192, 192), mean=(128, 128, 128), scale=True,
                 mirror=True, ignore_label=255, edge_weight = 1):
        self.root = root
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


        task_list = []
        scale_list = []
        image_path_list = []
        label_path_list = []
        tasks = glob.glob(os.path.join(self.root,'*'))

        for ki in range(len(tasks)):
            # if os.path.basename(tasks[ki]) != '5_3_ptc':
            #     continue
            tasks_id = os.path.basename(tasks[ki]).split('_')[0]
            scale_id = os.path.basename(tasks[ki]).split('_')[1]
            stain_folders = glob.glob(os.path.join(tasks[ki],'*'))

            for si in range(len(stain_folders)):
                images = glob.glob(os.path.join(stain_folders[si],'*'))
                for ri in range(len(images)):
                    if 'mask' in images[ri]:
                        continue
                    else:
                        image_root = images[ri]
                        _, ext = os.path.splitext(images[ri])

                        mask_root = glob.glob(os.path.join(stain_folders[si],os.path.basename(image_root).replace(ext,'_mask*')))[0]
                        # mask_root = 'aaa'

                    # print(os.path.join(stain_folders[si],os.path.basename(image_root).replace(ext,'_mask*')))
                        task_list.append(int(tasks_id))
                        scale_list.append(int(scale_id))
                        image_path_list.append(image_root)
                        label_path_list.append(mask_root)

        self.files = []
        self.df = pd.DataFrame(columns = ['image_path', 'label_path', 'name', 'task_id', 'scale_id'])

        print("Start preprocessing....")
        for i in range(len(image_path_list)):
            print(i)
            #print(image_path_list[i] + ', ' + str(task_list[i]) + ', ' + str(scale_list[i]))
            image_path = image_path_list[i]
            label_path = label_path_list[i]
            task_id = task_list[i]
            scale_id = scale_list[i]
            name = image_path.replace('/', '-')
            # name = osp.basename(image_path)
            img_file = image_path
            label_file = label_path
            #label = plt.imread(label_file)
            label = np.ones((512,512))

            boud_h, boud_w = np.where(label >= 1)
            self.df.loc[i] = [image_path, label_path, name, task_id, scale_id]

        self.df.to_csv(os.path.join(root, 'data_list.csv'), index = False)
        print('{} images are loaded!'.format(len(image_path_list)))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        # read png file
        image = plt.imread(datafiles["image"])
        label = plt.imread(datafiles["label"])

        name = datafiles["name"]
        task_id = datafiles["task_id"]
        scale_id = datafiles["scale_id"]

        # data augmentation
        image = image[:,:,:3]
        label = label[:,:,:3]

        image = np.expand_dims(image, axis=0)
        label = np.expand_dims(label, axis=0)

        # image = (image * 255).astype(np.uint8)
        # # image = self.image_aug_256(image)
        # image = image.astype(np.float32) / 255

        seed = np.random.rand(4)

        if seed[0] > 0.5:
            image, label = self.image_mask_aug(images=image, heatmaps=label)

        if seed[1] > 0.5:
            image = self.image_aug_color(images=image)

        if seed[2] > 0.5:
            image = self.image_aug_noise(images=image)

        # if task_id == 5:
        #     if seed[3] > 0.5:
        #         image = self.image_aug_resolution(images=image)

        label[label >= 0.5] = 1.
        label[label < 0.5] = 0.
        # weight[weight >= 0.5] = 1.
        # weight[weight < 0.5] = 0.

        # image = image.transpose((3, 1, 2, 0))  # Channel x H x W
        # label = label[:,:,:,0].transpose((1, 2, 0))

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

    trainset_dir = 'KI_data_testingset_demo'
    train_list = 'KI_data_testingset_demo'
    itrs_each_epoch = 250
    batch_size = 1
    input_size = (256,256)
    random_scale = False
    random_mirror = False

    save_img = '/media/dengr/Data2/KI_data_test_patches'
    save_mask = '/media/dengr/Data2/KI_data_test_patches'

    img_scale = 0.5

    trainloader = DataLoader(
        MOTSDataSet(trainset_dir, train_list, max_iters=itrs_each_epoch * batch_size,
                    crop_size=input_size, scale=random_scale, mirror=random_mirror),batch_size = 1, shuffle = False, num_workers = 8)

    for iter, batch in enumerate(trainloader):
        print(iter)
