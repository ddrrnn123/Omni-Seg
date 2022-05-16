import cv2 as cv2
import numpy as np
from PIL import Image
import os
import SimpleITK as sitk

from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
from skimage.transform import resize
import glob
import openslide
import matplotlib.pyplot as plt
import xmltodict
import pandas as pd
import math

from skimage.transform import rescale, resize


if __name__ == "__main__":

    data_dir = 'V11M25-279/20X'
    # data_dir = '/Data2/HumanKidney/Mouse_Atubular_Segmentation/test'
    svs_folder = '/Data2/HumanKidney/2profile/circlenet/scn'
    output_dir = 'clinical_patches'

    sections = glob.glob(os.path.join(data_dir, '*'))

    sections.sort()

    for si in range(len(sections)):
        name = os.path.basename(sections[si])
        print(name)

        output_folder = sections[si].replace(data_dir, output_dir).replace('.png', '')
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        img = plt.imread(sections[si])[:,:,:3]

        patch_size = 1024
        stride_size = 512

        padding_size = patch_size - stride_size

        img_padding = np.ones((img.shape[0] + 2 * padding_size, img.shape[1] + 2 * padding_size, 3)) * 220 / 255

        img_padding[padding_size:-padding_size, padding_size:-padding_size,:] = img

        img = img_padding

        stride_x = int(img.shape[0] / stride_size) - 1
        stride_y = int(img.shape[1] / stride_size) - 1

        for xi in range(stride_x):
            for yi in range(stride_y):

                x_ind = int(xi * stride_size)
                y_ind = int(yi * stride_size)

                now_patch = img[x_ind:x_ind + patch_size, y_ind:y_ind + patch_size, :]
                patch_dir = os.path.join(output_folder, '%d_%d.png' % (x_ind, y_ind))

                if now_patch.mean() < 220. / 255:
                    plt.imsave(patch_dir, now_patch)

