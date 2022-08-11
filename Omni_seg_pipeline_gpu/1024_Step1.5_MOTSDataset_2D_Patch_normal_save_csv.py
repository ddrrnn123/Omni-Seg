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

        cases = glob.glob(os.path.join(self.root,'*'))

        for ki in range(len(cases)):
            task_list = []
            scale_list = []
            image_path_list = []
            label_path_list = []

            case_name = os.path.basename(cases[ki])
            images = glob.glob(os.path.join(cases[ki],'*'))
            for ri in range(len(images)):
                if 'mask' in images[ri]:
                    continue
                else:
                    image_root = images[ri]
                    _, ext = os.path.splitext(images[ri])

                    # mask_root = glob.glob(os.path.join(stain_folders[si],os.path.basename(image_root).replace(ext,'_mask*')))[0]
                    mask_root = 'aaa'

                    task_list.append(int(0))
                    scale_list.append(int(0))
                    image_path_list.append(image_root)
                    label_path_list.append(case_name)

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

            self.df.to_csv(os.path.join(cases[ki], 'data_list.csv'), index = False)
            print('{} images are loaded!'.format(len(image_path_list)))

    def __len__(self):
        return len(self.files)

    def truncate(self, CT, task_id):
        min_HU = -325
        max_HU = 325
        subtract = 0
        divide = 325.

        # truncate
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU
        CT = CT - subtract
        CT = CT / divide
        return CT

    def id2trainId(self, label, task_id):
        if task_id == 0 or task_id == 1 or task_id == 3:
            organ = (label >= 1)
            tumor = (label == 2)
        elif task_id == 2:
            organ = (label == 1)
            tumor = (label == 2)
        elif task_id == 4 or task_id == 5:
            organ = None
            tumor = (label == 1)
        elif task_id == 6:
            organ = (label == 1)
            tumor = None
        else:
            print("Error, No such task!")
            return None

        shape = label.shape
        results_map = np.zeros((2, shape[0], shape[1], shape[2])).astype(np.float32)

        if organ is None:
            results_map[0, :, :, :] = results_map[0, :, :, :] - 1
        else:
            results_map[0, :, :, :] = np.where(organ, 1, 0)
        if tumor is None:
            results_map[1, :, :, :] = results_map[1, :, :, :] - 1
        else:
            results_map[1, :, :, :] = np.where(tumor, 1, 0)

        return results_map

    def locate_bbx(self, label, scaler, bbx):

        scale_d = int(self.crop_d * scaler)
        scale_h = int(self.crop_h * scaler)
        scale_w = int(self.crop_w * scaler)

        img_h, img_w, img_d = label.shape
        # boud_h, boud_w, boud_d = np.where(label >= 1)
        boud_h, boud_w, boud_d = bbx
        margin = 32  # pixels

        bbx_h_min = boud_h.min()
        bbx_h_max = boud_h.max()
        bbx_w_min = boud_w.min()
        bbx_w_max = boud_w.max()
        bbx_d_min = boud_d.min()
        bbx_d_max = boud_d.max()
        if (bbx_h_max - bbx_h_min) <= scale_h:
            bbx_h_maxt = bbx_h_max + math.ceil((scale_h - (bbx_h_max - bbx_h_min)) / 2)
            bbx_h_mint = bbx_h_min - math.ceil((scale_h - (bbx_h_max - bbx_h_min)) / 2)
            if bbx_h_mint < 0:
                bbx_h_maxt -= bbx_h_mint
                bbx_h_mint = 0
            bbx_h_max = bbx_h_maxt
            bbx_h_min = bbx_h_mint
        if (bbx_w_max - bbx_w_min) <= scale_w:
            bbx_w_maxt = bbx_w_max + math.ceil((scale_w - (bbx_w_max - bbx_w_min)) / 2)
            bbx_w_mint = bbx_w_min - math.ceil((scale_w - (bbx_w_max - bbx_w_min)) / 2)
            if bbx_w_mint < 0:
                bbx_w_maxt -= bbx_w_mint
                bbx_w_mint = 0
            bbx_w_max = bbx_w_maxt
            bbx_w_min = bbx_w_mint
        if (bbx_d_max - bbx_d_min) <= scale_d:
            bbx_d_maxt = bbx_d_max + math.ceil((scale_d - (bbx_d_max - bbx_d_min)) / 2)
            bbx_d_mint = bbx_d_min - math.ceil((scale_d - (bbx_d_max - bbx_d_min)) / 2)
            if bbx_d_mint < 0:
                bbx_d_maxt -= bbx_d_mint
                bbx_d_mint = 0
            bbx_d_max = bbx_d_maxt
            bbx_d_min = bbx_d_mint
        bbx_h_min = np.max([bbx_h_min - margin, 0])
        bbx_h_max = np.min([bbx_h_max + margin, img_h])
        bbx_w_min = np.max([bbx_w_min - margin, 0])
        bbx_w_max = np.min([bbx_w_max + margin, img_w])
        bbx_d_min = np.max([bbx_d_min - margin, 0])
        bbx_d_max = np.min([bbx_d_max + margin, img_d])

        if random.random() < 0.8:
            d0 = random.randint(bbx_d_min, np.max([bbx_d_max - scale_d, bbx_d_min]))
            h0 = random.randint(bbx_h_min, np.max([bbx_h_max - scale_h, bbx_h_min]))
            w0 = random.randint(bbx_w_min, np.max([bbx_w_max - scale_w, bbx_w_min]))
        else:
            d0 = random.randint(0, img_d - scale_d)
            h0 = random.randint(0, img_h - scale_h)
            w0 = random.randint(0, img_w - scale_w)
        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w
        return [h0, h1, w0, w1, d0, d1]

    def pad_image(self, img, target_size):
        """Pad an image up to the target size."""
        rows_missing = math.ceil(target_size[0] - img.shape[0])
        cols_missing = math.ceil(target_size[1] - img.shape[1])
        dept_missing = math.ceil(target_size[2] - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

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

        task_list = []
        scale_list = []
        image_path_list = []
        label_path_list = []
        tasks = glob.glob(os.path.join(self.root,'*'))
        for ki in range(len(tasks)):
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
                        print(image_root)
                        _, ext = os.path.splitext(images[ri])
                        mask_root = glob.glob(os.path.join(stain_folders[si],os.path.basename(image_root).replace(ext,'_mask*')))[0]
                        task_list.append(int(tasks_id))
                        scale_list.append(int(scale_id))
                        image_path_list.append(image_root)
                        label_path_list.append(mask_root)

        self.files = []

        for i in range(len(image_path_list)):
            print(image_path_list[i] + ', ' + label_path_list[i] )
            image_path = image_path_list[i]
            label_path = label_path_list[i]
            task_id = task_list[i]
            scale_id = scale_list[i]
            # if task_id != 1:
            #     name = osp.splitext(osp.basename(label_path))[0]
            # else:
            #name = osp.basename(label_path)
            name = image_path.replace('/','-')# osp.basename(label_path)
            img_file = image_path
            label_file = label_path

            self.files.append({
                "image": img_file,
                "label": label_file,
                "name": name,
                "task_id": task_id,
                "scale_id": scale_id,
            })

        print('{} images are loaded!'.format(len(image_path_list)))

    def __len__(self):
        return len(self.files)

    def truncate(self, CT, task_id):
        min_HU = -325
        max_HU = 325
        subtract = 0
        divide = 325.
        
        # truncate
        CT[np.where(CT <= min_HU)] = min_HU
        CT[np.where(CT >= max_HU)] = max_HU
        CT = CT - subtract
        CT = CT / divide
        return CT

    def id2trainId(self, label, task_id):
        if task_id == 0 or task_id == 1 or task_id == 3:
            organ = (label >= 1)
            tumor = (label == 2)
        elif task_id == 2:
            organ = (label == 1)
            tumor = (label == 2)
        elif task_id == 4 or task_id == 5:
            organ = None
            tumor = (label == 1)
        elif task_id == 6:
            organ = (label == 1)
            tumor = None
        else:
            print("Error, No such task!")
            return None

        shape = label.shape
        results_map = np.zeros((2, shape[0], shape[1], shape[2])).astype(np.float32)

        if organ is None:
            results_map[0, :, :, :] = results_map[0, :, :, :] - 1
        else:
            results_map[0, :, :, :] = np.where(organ, 1, 0)
        if tumor is None:
            results_map[1, :, :, :] = results_map[1, :, :, :] - 1
        else:
            results_map[1, :, :, :] = np.where(tumor, 1, 0)

        return results_map

    def locate_bbx(self, label, scaler):

        scale_d = int(self.crop_d * scaler)
        scale_h = int(self.crop_h * scaler)
        scale_w = int(self.crop_w * scaler)

        img_h, img_w, img_d = label.shape
        boud_h, boud_w, boud_d = np.where(label >= 1)
        margin = 32  # pixels
        bbx_h_min = boud_h.min()
        bbx_h_max = boud_h.max()
        bbx_w_min = boud_w.min()
        bbx_w_max = boud_w.max()
        bbx_d_min = boud_d.min()
        bbx_d_max = boud_d.max()
        if (bbx_h_max - bbx_h_min) <= scale_h:
            bbx_h_maxt = bbx_h_max + (scale_h - (bbx_h_max - bbx_h_min)) // 2
            bbx_h_mint = bbx_h_min - (scale_h - (bbx_h_max - bbx_h_min)) // 2
            bbx_h_max = bbx_h_maxt
            bbx_h_min = bbx_h_mint
        if (bbx_w_max - bbx_w_min) <= scale_w:
            bbx_w_maxt = bbx_w_max + (scale_w - (bbx_w_max - bbx_w_min)) // 2
            bbx_w_mint = bbx_w_min - (scale_w - (bbx_w_max - bbx_w_min)) // 2
            bbx_w_max = bbx_w_maxt
            bbx_w_min = bbx_w_mint
        if (bbx_d_max - bbx_d_min) <= scale_d:
            bbx_d_maxt = bbx_d_max + (scale_d - (bbx_d_max - bbx_d_min)) // 2
            bbx_d_mint = bbx_d_min - (scale_d - (bbx_d_max - bbx_d_min)) // 2
            bbx_d_max = bbx_d_maxt
            bbx_d_min = bbx_d_mint
        bbx_h_min = np.max([bbx_h_min - margin, 0])
        bbx_h_max = np.min([bbx_h_max + margin, img_h])
        bbx_w_min = np.max([bbx_w_min - margin, 0])
        bbx_w_max = np.min([bbx_w_max + margin, img_w])
        bbx_d_min = np.max([bbx_d_min - margin, 0])
        bbx_d_max = np.min([bbx_d_max + margin, img_d])

        if random.random() < 0.8:
            d0 = random.randint(bbx_d_min, bbx_d_max - scale_d)
            h0 = random.randint(bbx_h_min, bbx_h_max - scale_h)
            w0 = random.randint(bbx_w_min, bbx_w_max - scale_w)
        else:
            d0 = random.randint(0, img_d - scale_d)
            h0 = random.randint(0, img_h - scale_h)
            w0 = random.randint(0, img_w - scale_w)
        d1 = d0 + scale_d
        h1 = h0 + scale_h
        w1 = w0 + scale_w
        return [h0, h1, w0, w1, d0, d1]

    def pad_image(self, img, target_size):
        """Pad an image up to the target size."""
        rows_missing = math.ceil(target_size[0] - img.shape[0])
        cols_missing = math.ceil(target_size[1] - img.shape[1])
        dept_missing = math.ceil(target_size[2] - img.shape[2])
        if rows_missing < 0:
            rows_missing = 0
        if cols_missing < 0:
            cols_missing = 0
        if dept_missing < 0:
            dept_missing = 0

        padded_img = np.pad(img, ((0, rows_missing), (0, cols_missing), (0, dept_missing)), 'constant')
        return padded_img

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

        #image, label = self.image_mask_aug(images=image, heatmaps=label)
        #image = self.image_aug(images=image)

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

# def get_train_transform():
#     tr_transforms = []
#
#     tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key="image"))
#     tr_transforms.append(
#         GaussianBlurTransform(blur_sigma=(0.5, 1.), different_sigma_per_channel=True, p_per_channel=0.5,
#                               p_per_sample=0.2, data_key="image"))
#     tr_transforms.append(BrightnessMultiplicativeTransform((0.75, 1.25), p_per_sample=0.15, data_key="image"))
#     tr_transforms.append(BrightnessTransform(0.0, 0.1, True, p_per_sample=0.15, p_per_channel=0.5, data_key="image"))
#     tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key="image"))
#     tr_transforms.append(
#         SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5, order_downsample=0,
#                                        order_upsample=3, p_per_sample=0.25,
#                                        ignore_axes=None, data_key="image"))
#     tr_transforms.append(GammaTransform(gamma_range=(0.7, 1.5), invert_image=False, per_channel=True, retain_stats=True,
#                                         p_per_sample=0.15, data_key="image"))
#
#     # now we compose these transforms together
#     tr_transforms = Compose(tr_transforms)
#     return tr_transforms


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

    docker = 1

    if docker:
        trainset_dir = '/desktop/src/extra/OmniSeg_MouthKidney_Pipeline/clinical_patches'
        train_list = '/desktop/src/extra/OmniSeg_MouthKidney_Pipeline/clinical_patches'
    else:
        trainset_dir = '/home/lengh2/Desktop/Haoju_Leng/DockerFiles/test/src/extra/OmniSeg_MouthKidney_Pipeline/clinical_patches'
        train_list = '/home/lengh2/Desktop/Haoju_Leng/DockerFiles/test/src/extra/OmniSeg_MouthKidney_Pipeline/clinical_patches'

    itrs_each_epoch = 250
    batch_size = 1
    input_size = (256,256)
    random_scale = False
    random_mirror = False

    save_img = '/KI_data_test_patches'
    save_mask = '/KI_data_test_patches'

    img_scale = 0.5

    trainloader = DataLoader(
        MOTSDataSet(trainset_dir, train_list, max_iters=itrs_each_epoch * batch_size,
                    crop_size=input_size, scale=random_scale, mirror=random_mirror),batch_size = 1, shuffle = False, num_workers = 8)

    for iter, batch in enumerate(trainloader):
        print(iter)
