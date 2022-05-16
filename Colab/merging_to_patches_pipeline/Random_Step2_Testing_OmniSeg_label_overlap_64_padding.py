import argparse
import os, sys
import pandas as pd

sys.path.append("/Data/DoDNet/")
from skimage.transform import rescale, resize
import glob
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from scipy.ndimage import morphology
from matplotlib import cm
from torchvision import transforms

import skimage
import re

import os.path as osp
# from MOTSDataset_2D_Patch_normal import MOTSDataSet, MOTSValDataSet, my_collate
from MOTSDataset_2D_Patch_joint_csv_PTC import MOTSValDataSet as MOTSValDataSet_joint

#from unet2D_ns import UNet2D as UNet2D_ns

import random
import timeit
from tensorboardX import SummaryWriter
import loss_functions.loss_2D as loss

from sklearn import metrics
from math import ceil

from engine import Engine
from apex import amp
from apex.parallel import convert_syncbn_model
#from focalloss import FocalLoss2dff
from sklearn.metrics import f1_score, confusion_matrix

start = timeit.default_timer()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from util.image_pool import ImagePool

from unet2D_Dodnet_ns import UNet2D as unet2D_ns

def one_hot_2D(targets,C = 2):
    targets_extend=targets.clone()
    targets_extend.unsqueeze_(1) # convert to Nx1xHxW
    one_hot = torch.cuda.FloatTensor(targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3)).zero_()
    one_hot.scatter_(1, targets_extend, 1)
    return one_hot


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments(img, output_folder):

    parser = argparse.ArgumentParser(description="DeepLabV3")
    parser.add_argument("--trainset_dir", type=str, default='/Data2/KI_data_train_scale_aug_patch')

    parser.add_argument("--valset_dir", type=str, default=img + '/data_list.csv')
    parser.add_argument("--output_dir", type=str, default=output_folder)

    parser.add_argument("--snapshot_dir", type=str, default='snapshots_2D/fold1_with_white_UNet2D_ns_normalwhole_1106')
    parser.add_argument("--reload_path", type=str, default='snapshots_2D/fold1_with_white_UNet2D_ns_normalwhole_1106/MOTS_DynConv_fold1_with_white_UNet2D_ns_normalwhole_1106_e89.pth')
    parser.add_argument("--best_epoch", type=int, default=100)

    # parser.add_argument("--validsetname", type=str, default='scale')
    parser.add_argument("--validsetname", type=str, default='normal')
    #parser.add_argument("--valset_dir", type=str, default='/Data2/Demo_KI_data_train_patch_with_white')
    parser.add_argument("--train_list", type=str, default='list/MOTS/MOTS_train.txt')
    parser.add_argument("--val_list", type=str, default='list/MOTS/xx.txt')
    parser.add_argument("--edge_weight", type=float, default=1.2)
    # parser.add_argument("--snapshot_dir", type=str, default='1027results/fold1_with_white_Unet2D_scaleid3_fullydata_1027')
    parser.add_argument("--reload_from_checkpoint", type=str2bool, default=True)
    parser.add_argument("--input_size", type=str, default='256,256')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument("--FP16", type=str2bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--itrs_each_epoch", type=int, default=250)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--val_pred_every", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_std", type=str2bool, default=True)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--ignore_label", type=int, default=255)
    parser.add_argument("--is_training", action="store_true")
    parser.add_argument("--random_mirror", type=str2bool, default=True)
    parser.add_argument("--random_scale", type=str2bool, default=True)
    parser.add_argument("--random_seed", type=int, default=1234)
    parser.add_argument("--gpu", type=str, default='None')
    return parser


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, i_iter, lr, num_stemps, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(lr, i_iter, num_stemps, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr

def count_score_only_two(preds, labels, rmin, rmax, cmin, cmax):

    Val_F1 = 0
    Val_DICE = 0
    Val_TPR = 0
    Val_PPV = 0
    cnt = 0

    for ki in range(len(preds)):
        cnt += 1
        pred = preds[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]
        label = labels[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]

        Val_DICE += dice_score(pred, label)
        #preds1 = preds[:,1,...].detach().view(-1).cpu().numpy()
        preds1 = pred[1,...].flatten().detach().cpu().numpy()
        #labels1 = labels[:,1,...].view(-1).cpu().numpy()
        labels1 = label[1,...].detach().flatten().detach().cpu().numpy()

        Val_F1 += f1_score(preds1, labels1, average='macro')

    return Val_F1/cnt, Val_DICE/cnt, 0., 0.


def surfd(input1, input2, sampling=1, connectivity=1):
    # input_1 = np.atleast_1d(input1.astype(bool))
    # input_2 = np.atleast_1d(input2.astype(bool))

    conn = morphology.generate_binary_structure(input1.ndim, connectivity)

    S = input1 - morphology.binary_erosion(input1, conn)
    Sprime = input2 - morphology.binary_erosion(input2, conn)

    S = np.atleast_1d(S.astype(bool))
    Sprime = np.atleast_1d(Sprime.astype(bool))


    dta = morphology.distance_transform_edt(~S, sampling)
    dtb = morphology.distance_transform_edt(~Sprime, sampling)

    sds = np.concatenate([np.ravel(dta[Sprime != 0]), np.ravel(dtb[S != 0])])

    return np.max(sds), np.mean(sds)

def count_score(preds, labels, rmin, rmax, cmin, cmax):

    Val_F1 = 0
    Val_DICE = 0
    Val_HD = 0
    Val_MSD = 0
    cnt = 0

    for ki in range(len(preds)):
        cnt += 1
        pred = preds[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]
        label = labels[ki,:,rmin[ki]:rmax[ki],cmin[ki]:cmax[ki]]

        Val_DICE += dice_score(pred, label)
        #preds1 = preds[:,1,...].detach().view(-1).cpu().numpy()
        preds0 = pred[1, ...].detach().cpu().numpy()
        labels0 = label[1, ...].detach().detach().cpu().numpy()

        preds1 = pred[1,...].flatten().detach().cpu().numpy()
        #labels1 = labels[:,1,...].view(-1).cpu().numpy()
        labels1 = label[1,...].detach().flatten().detach().cpu().numpy()

        try:

            hausdorff, meansurfaceDistance = surfd(preds0, labels0)
            Val_HD += hausdorff
            Val_MSD += meansurfaceDistance

            Val_F1 += f1_score(preds1, labels1, average='macro')

        except:
            Val_DICE += 1.
            Val_F1 += 1.
            Val_HD += 0.
            Val_MSD += 0.

    return Val_F1/cnt, Val_DICE/cnt, Val_HD/cnt, Val_MSD/cnt

def dice_score(preds, labels):  # on GPU
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    predict = preds.contiguous().view(preds.shape[0], -1)
    target = labels.contiguous().view(labels.shape[0], -1)

    num = torch.sum(torch.mul(predict, target), dim=1)
    den = torch.sum(predict, dim=1) + torch.sum(target, dim=1) + 1

    dice = 2 * num / den

    return dice.mean()

def mask_to_box(tensor):
    tensor = tensor.permute([0,2,3,1]).cpu().numpy()
    rmin = np.zeros((4))
    rmax = np.zeros((4))
    cmin = np.zeros((4))
    cmax = np.zeros((4))
    for ki in range(len(tensor)):
        rows = np.any(tensor[ki], axis=1)
        cols = np.any(tensor[ki], axis=0)

        rmin[ki], rmax[ki] = np.where(rows)[0][[0, -1]]
        cmin[ki], cmax[ki] = np.where(cols)[0][[0, -1]]

    # plt.imshow(tensor[0,int(rmin[0]):int(rmax[0]),int(cmin[0]):int(cmax[0]),:])
    return rmin.astype(np.uint32), rmax.astype(np.uint32), cmin.astype(np.uint32), cmax.astype(np.uint32)



def testing_10X(imgs_10X, now_task, now_scale, volumeName, patch_size, batch_size, model, output_folder, imgs):

    stride = int(patch_size / 4)
    merge_edge = 64
    padding_size = patch_size - merge_edge - stride

    'Do the padding on patch'
    imgs_10X_padding = torch.zeros((imgs_10X.shape[0], imgs_10X.shape[1], imgs_10X.shape[2] + padding_size * 2,
                                   imgs_10X.shape[3] + padding_size * 2))
    imgs_10X_padding[:, :, padding_size:-padding_size, padding_size: - padding_size] = imgs_10X

    imgs_10X = imgs_10X_padding

    grid_num = int(( imgs_10X.shape[2] - patch_size) /  stride) + 1
    all_num = grid_num * grid_num

    img_batch = torch.zeros((all_num, 3, patch_size, patch_size))
    cnt = 0
    majority = 8


    x_list = np.zeros((all_num))
    y_list = np.zeros((all_num))


    for ki in range(grid_num):
        for kj in range(grid_num):
            if ki == 0 or ki == (grid_num - 1):
                start_x = int(ki * stride)
            else:
                num = np.random.randint(0, 2 * stride, 1)
                start_x = int(ki * stride + num - stride)

            if kj == 0 or kj == (grid_num - 1):
                start_y = int(kj * stride)
            else:
                num = np.random.randint(0, 2 * stride, 1)
                start_y = int(kj * stride + num - stride)

            end_x = start_x + patch_size
            end_y = start_y + patch_size

            x_list[cnt] = start_x
            y_list[cnt] = start_y

            img_batch[cnt] = imgs_10X[0, :, start_x:end_x, start_y:end_y]
            cnt += 1

    batch_num = int(all_num / batch_size) + 1

    preds_batch = torch.zeros((all_num, 2, patch_size, patch_size))
    for bi in range(batch_num):
        if bi != batch_num - 1:
            preds_batch[bi * batch_size : (bi + 1) * batch_size] = model(img_batch[bi * batch_size : (bi + 1) * batch_size].cuda(), torch.ones(batch_size).cuda() * now_task, torch.ones(batch_size).cuda() * now_scale)
        else:
            preds_batch[-batch_size:] = model(img_batch[-batch_size:].cuda(), torch.ones(batch_size).cuda() * now_task, torch.ones(batch_size).cuda() * now_scale)

    prediction = torch.zeros(imgs_10X.shape[2], imgs_10X.shape[3])

    cnt = 0
    for ki in range(grid_num):
        for kj in range(grid_num):

            start_x = int(x_list[cnt])
            start_y = int(y_list[cnt])

            end_x = start_x + patch_size
            end_y = start_y + patch_size

            # print(start_x, start_y)

            if 1:#ki != 0 and ki != 64 and kj != 0 and kj != 64:
                now_prediction = (preds_batch[cnt, 1] > preds_batch[cnt, 0]).detach().numpy().astype(np.float32)
                prediction[start_x + merge_edge:end_x - merge_edge, start_y + merge_edge:end_y - merge_edge] = prediction[start_x + merge_edge:end_x - merge_edge,
                                                                               start_y + merge_edge:end_y - merge_edge] + now_prediction[
                                                                                                          merge_edge:-merge_edge,
                                                                                                          merge_edge:-merge_edge]
                cnt += 1
                # plt.imsave(os.path.join(output_folder, 'Small_map_%d_%d.png' % (start_x, start_y)), now_prediction, cmap=cm.gray)

            else:
                now_prediction = (preds_batch[cnt, 1] > preds_batch[cnt, 0]).detach().numpy().astype(np.float32)
                prediction[start_x:end_x, start_y:end_y] = prediction[start_x:end_x, start_y:end_y] + now_prediction
                cnt += 1


    # prediction[merge_edge:-merge_edge, merge_edge:-merge_edge] = prediction[merge_edge:-merge_edge, merge_edge:-merge_edge] * (prediction[merge_edge:-merge_edge, merge_edge:-merge_edge] >= majority)
    # prediction[:merge_edge, merge_edge:-merge_edge] = prediction[:merge_edge, merge_edge:-merge_edge] * (prediction[:merge_edge, merge_edge:-merge_edge] >= majority_second)
    # prediction[-merge_edge:, merge_edge:-merge_edge] = prediction[-merge_edge:, merge_edge:-merge_edge] * (prediction[-merge_edge:, merge_edge:-merge_edge] >= majority_second)
    # prediction[merge_edge:-merge_edge, :merge_edge] = prediction[merge_edge:-merge_edge, :merge_edge] * (prediction[merge_edge:-merge_edge, :merge_edge] >= majority_second)
    # prediction[merge_edge:-merge_edge, -merge_edge:] = prediction[merge_edge:-merge_edge, -merge_edge:] * (prediction[merge_edge:-merge_edge, -merge_edge:] >= majority_second)

    prediction = prediction >= majority

    prediction = prediction[padding_size:-padding_size, padding_size:-padding_size]

    resize_function = transforms.Resize(1024)
    prediction = resize_function(prediction.unsqueeze(0)).squeeze(0)
    big_img_resize = imgs[0]

    out_image = big_img_resize.permute([1, 2, 0]).detach().cpu().numpy()
    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())

    plt.imsave(os.path.join(output_folder, 'Big_map.png'), img)
    plt.imsave(os.path.join(output_folder, 'Big_pred.png'),prediction, cmap=cm.gray)

def testing_5X(imgs_5X, now_task, now_scale, volumeName, patch_size, batch_size, model, output_folder, imgs):
    stride = int(patch_size / 4)
    merge_edge = 64
    padding_size = patch_size - merge_edge - stride

    'Do the padding on patch'
    imgs_5X_padding = torch.zeros((imgs_5X.shape[0], imgs_5X.shape[1], imgs_5X.shape[2] + padding_size * 2,
                                   imgs_5X.shape[3] + padding_size * 2))
    imgs_5X_padding[:, :, padding_size:-padding_size, padding_size: - padding_size] = imgs_5X

    imgs_5X = imgs_5X_padding

    grid_num = int(( imgs_5X.shape[2] - patch_size) /  stride) + 1
    all_num = grid_num * grid_num

    img_batch = torch.zeros((all_num, 3, patch_size, patch_size))
    cnt = 0
    majority = 8


    x_list = np.zeros((all_num))
    y_list = np.zeros((all_num))

    for ki in range(grid_num):
        for kj in range(grid_num):
            if ki == 0 or ki == (grid_num - 1):
                start_x = int(ki * stride)
            else:
                num = np.random.randint(0, 2 * stride, 1)
                start_x = int(ki * stride + num - stride)

            if kj == 0 or kj == (grid_num - 1):
                start_y = int(kj * stride)
            else:
                num = np.random.randint(0, 2 * stride, 1)
                start_y = int(kj * stride + num - stride)

            end_x = start_x + patch_size
            end_y = start_y + patch_size

            x_list[cnt] = start_x
            y_list[cnt] = start_y

            img_batch[cnt] = imgs_5X[0, :, start_x:end_x, start_y:end_y]
            cnt += 1

    batch_num = int(all_num / batch_size) + 1

    preds_batch = torch.zeros((all_num, 2, patch_size, patch_size))
    for bi in range(batch_num):
        if bi != batch_num - 1:
            preds_batch[bi * batch_size: (bi + 1) * batch_size] = model(
                img_batch[bi * batch_size: (bi + 1) * batch_size].cuda(), torch.ones(batch_size).cuda() * now_task,
                torch.ones(batch_size).cuda() * now_scale)
        else:
            preds_batch[-batch_size:] = model(img_batch[-batch_size:].cuda(), torch.ones(batch_size).cuda() * now_task,
                                              torch.ones(batch_size).cuda() * now_scale)

    prediction = torch.zeros(imgs_5X.shape[2], imgs_5X.shape[3])

    cnt = 0
    for ki in range(grid_num):
        for kj in range(grid_num):

            start_x = int(x_list[cnt])
            start_y = int(y_list[cnt])

            end_x = start_x + patch_size
            end_y = start_y + patch_size

            # print(start_x, start_y)

            if 1:  # ki != 0 and ki != 64 and kj != 0 and kj != 64:
                now_prediction = (preds_batch[cnt, 1] > preds_batch[cnt, 0]).detach().numpy().astype(np.float32)
                prediction[start_x + merge_edge:end_x - merge_edge,
                start_y + merge_edge:end_y - merge_edge] = prediction[start_x + merge_edge:end_x - merge_edge,
                                                           start_y + merge_edge:end_y - merge_edge] + now_prediction[
                                                                                                      merge_edge:-merge_edge,
                                                                                                      merge_edge:-merge_edge]
                cnt += 1
                # plt.imsave(os.path.join(output_folder, 'Small_map_%d_%d.png' % (start_x, start_y)), now_prediction, cmap=cm.gray)

            else:
                now_prediction = (preds_batch[cnt, 1] > preds_batch[cnt, 0]).detach().numpy().astype(np.float32)
                prediction[start_x:end_x, start_y:end_y] = prediction[start_x:end_x, start_y:end_y] + now_prediction
                cnt += 1

    # prediction[merge_edge:-merge_edge, merge_edge:-merge_edge] = prediction[merge_edge:-merge_edge,
    #                                                              merge_edge:-merge_edge] * (
    #                                                                          prediction[merge_edge:-merge_edge,
    #                                                                          merge_edge:-merge_edge] >= majority)
    # prediction[:merge_edge, merge_edge:-merge_edge] = prediction[:merge_edge, merge_edge:-merge_edge] * (
    #             prediction[:merge_edge, merge_edge:-merge_edge] >= majority_second)
    # prediction[-merge_edge:, merge_edge:-merge_edge] = prediction[-merge_edge:, merge_edge:-merge_edge] * (
    #             prediction[-merge_edge:, merge_edge:-merge_edge] >= majority_second)
    # prediction[merge_edge:-merge_edge, :merge_edge] = prediction[merge_edge:-merge_edge, :merge_edge] * (
    #             prediction[merge_edge:-merge_edge, :merge_edge] >= majority_second)
    # prediction[merge_edge:-merge_edge, -merge_edge:] = prediction[merge_edge:-merge_edge, -merge_edge:] * (
    #             prediction[merge_edge:-merge_edge, -merge_edge:] >= majority_second)
    #
    # prediction = prediction > 0
    #
    # prediction = prediction[3 * stride:-3 * stride, 3 * stride:-3 * stride]

    prediction = prediction >= majority

    if prediction.sum() > 0.2 * prediction.shape[0] * prediction.shape[1]:
        prediction = np.zeros((prediction.shape))

    prediction = prediction[padding_size:-padding_size, padding_size:-padding_size]

    resize_function = transforms.Resize(1024)
    prediction = resize_function(prediction.unsqueeze(0)).squeeze(0)
    big_img_resize = imgs[0]

    out_image = big_img_resize.permute([1, 2, 0]).detach().cpu().numpy()
    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())

    plt.imsave(os.path.join(output_folder, 'Big_map.png'), img)
    plt.imsave(os.path.join(output_folder, 'Big_pred.png'), prediction, cmap=cm.gray)


def testing_40X(imgs_40X, now_task, now_scale, volumeName, patch_size, batch_size, model, output_folder, imgs):
    stride = int(patch_size / 4)
    merge_edge = 64
    padding_size = patch_size - merge_edge - stride

    'Do the padding on patch'
    imgs_40X_padding = torch.zeros((imgs_40X.shape[0], imgs_40X.shape[1], imgs_40X.shape[2] + padding_size * 2,
                                   imgs_40X.shape[3] + padding_size * 2))
    imgs_40X_padding[:, :, padding_size:-padding_size, padding_size: - padding_size] = imgs_40X

    imgs_40X = imgs_40X_padding

    grid_num = int(( imgs_40X.shape[2] - patch_size) /  stride) + 1
    all_num = grid_num * grid_num

    img_batch = torch.zeros((all_num, 3, patch_size, patch_size))
    cnt = 0
    majority = 8


    x_list = np.zeros((all_num))
    y_list = np.zeros((all_num))

    for ki in range(grid_num):
        for kj in range(grid_num):
            if ki == 0 or ki == (grid_num - 1):
                start_x = int(ki * stride)
            else:
                num = np.random.randint(0, 2 * stride, 1)
                start_x = int(ki * stride + num - stride)

            if kj == 0 or kj == (grid_num - 1):
                start_y = int(kj * stride)
            else:
                num = np.random.randint(0, 2 * stride, 1)
                start_y = int(kj * stride + num - stride)

            end_x = start_x + patch_size
            end_y = start_y + patch_size

            x_list[cnt] = start_x
            y_list[cnt] = start_y

            img_batch[cnt] = imgs_40X[0, :, start_x:end_x, start_y:end_y]
            cnt += 1

    batch_num = int(all_num / batch_size) + 1

    preds_batch = torch.zeros((all_num, 2, patch_size, patch_size))
    for bi in range(batch_num):
        if bi != batch_num - 1:
            preds_batch[bi * batch_size: (bi + 1) * batch_size] = model(
                img_batch[bi * batch_size: (bi + 1) * batch_size].cuda(), torch.ones(batch_size).cuda() * now_task,
                torch.ones(batch_size).cuda() * now_scale)
        else:
            preds_batch[-batch_size:] = model(img_batch[-batch_size:].cuda(), torch.ones(batch_size).cuda() * now_task,
                                              torch.ones(batch_size).cuda() * now_scale)

    prediction = torch.zeros(imgs_40X.shape[2], imgs_40X.shape[3])

    cnt = 0
    for ki in range(grid_num):
        for kj in range(grid_num):

            start_x = int(x_list[cnt])
            start_y = int(y_list[cnt])

            end_x = start_x + patch_size
            end_y = start_y + patch_size

            # print(start_x, start_y)

            if 1:  # ki != 0 and ki != 64 and kj != 0 and kj != 64:
                now_prediction = (preds_batch[cnt, 1] > preds_batch[cnt, 0]).detach().numpy().astype(np.float32)
                prediction[start_x + merge_edge:end_x - merge_edge,
                start_y + merge_edge:end_y - merge_edge] = prediction[start_x + merge_edge:end_x - merge_edge,
                                                           start_y + merge_edge:end_y - merge_edge] + now_prediction[
                                                                                                      merge_edge:-merge_edge,
                                                                                                      merge_edge:-merge_edge]
                cnt += 1
                # plt.imsave(os.path.join(output_folder, 'Small_map_%d_%d.png' % (start_x, start_y)), now_prediction, cmap=cm.gray)

            else:
                now_prediction = (preds_batch[cnt, 1] > preds_batch[cnt, 0]).detach().numpy().astype(np.float32)
                prediction[start_x:end_x, start_y:end_y] = prediction[start_x:end_x, start_y:end_y] + now_prediction
                cnt += 1

    # prediction[merge_edge:-merge_edge, merge_edge:-merge_edge] = prediction[merge_edge:-merge_edge,
    #                                                              merge_edge:-merge_edge] * (
    #                                                                          prediction[merge_edge:-merge_edge,
    #                                                                          merge_edge:-merge_edge] >= majority)
    # prediction[:merge_edge, merge_edge:-merge_edge] = prediction[:merge_edge, merge_edge:-merge_edge] * (
    #             prediction[:merge_edge, merge_edge:-merge_edge] >= majority_second)
    # prediction[-merge_edge:, merge_edge:-merge_edge] = prediction[-merge_edge:, merge_edge:-merge_edge] * (
    #             prediction[-merge_edge:, merge_edge:-merge_edge] >= majority_second)
    # prediction[merge_edge:-merge_edge, :merge_edge] = prediction[merge_edge:-merge_edge, :merge_edge] * (
    #             prediction[merge_edge:-merge_edge, :merge_edge] >= majority_second)
    # prediction[merge_edge:-merge_edge, -merge_edge:] = prediction[merge_edge:-merge_edge, -merge_edge:] * (
    #             prediction[merge_edge:-merge_edge, -merge_edge:] >= majority_second)
    #
    # prediction = prediction > 0
    #
    # prediction = prediction[3 * stride:-3 * stride, 3 * stride:-3 * stride]

    prediction = prediction >= majority

    prediction = prediction[padding_size:-padding_size, padding_size:-padding_size]

    resize_function = transforms.Resize(1024)
    prediction = resize_function(prediction.unsqueeze(0)).squeeze(0)
    big_img_resize = imgs[0]

    out_image = big_img_resize.permute([1, 2, 0]).detach().cpu().numpy()
    img = (out_image - out_image.min()) / (out_image.max() - out_image.min())

    plt.imsave(os.path.join(output_folder, 'Big_map.png'), img)
    plt.imsave(os.path.join(output_folder, 'Big_pred.png'), prediction, cmap=cm.gray)


def main(img, output_dir, case_name):
    """Create the model and start the training."""

    output_folder = os.path.join(output_dir, case_name)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    parser = get_arguments(img, output_folder)
    print(parser)

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()
        if args.num_gpus > 1:
            torch.cuda.set_device(args.local_rank)

        writer = SummaryWriter(args.snapshot_dir)

        if not args.gpu == 'None':
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)

        cudnn.benchmark = True
        seed = args.random_seed
        if engine.distributed:
            seed = args.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        # Create model
        criterion = None
        model = unet2D_ns(num_classes=args.num_classes, weight_std = False)
        check_wo_gpu = 0

        print(f"total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")

        if not check_wo_gpu:
            device = torch.device('cuda:{}'.format(args.local_rank))
            model.to(device)

        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.99, nesterov=True)

        if not check_wo_gpu:
            if args.FP16:
                print("Note: Using FP16 during training************")
                model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

            if args.num_gpus > 1:
                model = engine.data_parallel(model)

        # load checkpoint...
        if args.reload_from_checkpoint:
            print('loading from checkpoint: {}'.format(args.reload_path))
            if os.path.exists(args.reload_path):
                if args.FP16:
                    checkpoint = torch.load(args.reload_path, map_location=torch.device('cpu'))
                    model.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    amp.load_state_dict(checkpoint['amp'])
                else:
                    model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))
            else:
                print('File not exists in the reload path: {}'.format(args.reload_path))

        if not check_wo_gpu:
            weights = [1., 10.]
            class_weights = torch.FloatTensor(weights).cuda()
            loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes).to(device)
            loss_seg_CE = loss.CELoss4MOTS(num_classes=args.num_classes, ignore_index=255).to(device)
            #criterion1 = nn.CrossEntropyLoss(weight=class_weights).to(device)
            #criterion2 = FocalLoss2d(weight=class_weights).to(device)

        else:
            weights = [1., 10.]
            class_weights = torch.FloatTensor(weights)
            loss_seg_DICE = loss.DiceLoss4MOTS(num_classes=args.num_classes)
            loss_seg_CE = loss.CELoss4MOTS(num_classes=args.num_classes, ignore_index=255)
            #criterion1 = nn.CrossEntropyLoss(weight=class_weights)
            #criterion2 = FocalLoss2d(weight=class_weights)

        if not os.path.exists(args.snapshot_dir):
            os.makedirs(args.snapshot_dir)

        edge_weight = args.edge_weight

        num_worker = 8

        valloader = DataLoader(
            MOTSValDataSet_joint(args.valset_dir, args.val_list, max_iters=args.itrs_each_epoch * args.batch_size,
                           crop_size=input_size, scale=args.random_scale, mirror=args.random_mirror,
                           edge_weight=edge_weight),batch_size=1,shuffle=False,num_workers=num_worker)

        all_tr_loss = []
        all_va_loss = []
        train_loss_MA = None
        val_loss_MA = None

        val_best_loss = 999999
        batch_size = args.batch_size

        model.load_state_dict(torch.load(args.reload_path, map_location=torch.device('cpu')))

        model.eval()

        with torch.no_grad():
            for iter, batch in enumerate(valloader):

                'dataloader'
                imgs = batch[0].cuda()
                lbls = batch[1].cuda()
                wt = batch[2].cuda().float()
                volumeName = batch[3][0]
                t_ids = batch[4].cuda()
                s_ids = batch[5]

                optimizer.zero_grad()

                transform_5X = transforms.Resize(256)
                transform_10X = transforms.Resize(512)

                imgs_5X = transform_5X(imgs)
                imgs_10X = transform_10X(imgs)
                imgs_40X = imgs

                patch_size = 256
                batch_size = 4

                check_output_folder = os.path.join(output_folder, volumeName.split('-')[-1].replace('.png', ''),'5_3_ptc')
                if os.path.exists(check_output_folder):
                    continue

                '0_dt'
                now_task = 0
                now_scale = 1

                now_output_folder = os.path.join(output_folder, volumeName.split('-')[-1].replace('.png', ''), '0_1_dt')

                if not os.path.exists(now_output_folder):
                    os.makedirs(now_output_folder)
                testing_10X(imgs_10X, now_task, now_scale, volumeName, patch_size, batch_size, model, now_output_folder, imgs)

                '1_pt'
                now_task = 1
                now_scale = 1

                now_output_folder = os.path.join(output_folder, volumeName.split('-')[-1].replace('.png', ''), '1_1_pt')
                if not os.path.exists(now_output_folder):
                    os.makedirs(now_output_folder)
                testing_10X(imgs_10X, now_task, now_scale, volumeName, patch_size, batch_size, model, now_output_folder, imgs)

                '2_cap'
                now_task = 2
                now_scale = 0

                now_output_folder = os.path.join(output_folder, volumeName.split('-')[-1].replace('.png', ''), '2_0_capsule')
                if not os.path.exists(now_output_folder):
                    os.makedirs(now_output_folder)
                testing_5X(imgs_5X, now_task, now_scale, volumeName, patch_size, batch_size, model, now_output_folder, imgs)

                '3_tuft'
                now_task = 3
                now_scale = 0

                now_output_folder = os.path.join(output_folder, volumeName.split('-')[-1].replace('.png', ''), '3_0_tuft')
                if not os.path.exists(now_output_folder):
                    os.makedirs(now_output_folder)
                testing_5X(imgs_5X, now_task, now_scale, volumeName, patch_size, batch_size, model, now_output_folder, imgs)


                '4_vessel'
                now_task = 4
                now_scale = 1
                now_output_folder = os.path.join(output_folder, volumeName.split('-')[-1].replace('.png', ''),'4_1_vessel')
                if not os.path.exists(now_output_folder):
                    os.makedirs(now_output_folder)
                testing_10X(imgs_10X, now_task, now_scale, volumeName, patch_size, batch_size, model, now_output_folder, imgs)


                '5_ptc'
                now_task = 5
                now_scale = 3

                now_output_folder = os.path.join(output_folder, volumeName.split('-')[-1].replace('.png', ''),'5_3_ptc')
                if not os.path.exists(now_output_folder):
                    os.makedirs(now_output_folder)
                testing_40X(imgs_40X, now_task, now_scale, volumeName, patch_size, batch_size, model, now_output_folder, imgs)


    end = timeit.default_timer()
    print(end - start, 'seconds')


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

if __name__ == '__main__':

    data_dir = 'clinical_patches/'
    output_dir = 'segmentation_merge/'

    cases = glob.glob(os.path.join(data_dir,'*'))
    cases.sort(key=natural_keys)

    for now_case in cases:
        case_name = os.path.basename(now_case)
        # for img in images:
        main(now_case, output_dir, case_name)