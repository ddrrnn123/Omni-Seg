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

def get_contour_detection(img, contour, cnt_Big, down_rate, shift):
    vertices = contour['Vertices']['Vertex']
    cnt = np.zeros((4,1,2))

    cnt[0, 0, 0] = vertices[1]['@X']
    cnt[0, 0, 1] = vertices[0]['@Y']
    cnt[1, 0, 0] = vertices[1]['@X']
    cnt[1, 0, 1] = vertices[1]['@Y']
    cnt[2, 0, 0] = vertices[0]['@X']
    cnt[2, 0, 1] = vertices[1]['@Y']
    cnt[3, 0, 0] = vertices[0]['@X']
    cnt[3, 0, 1] = vertices[0]['@Y']

    cnt = cnt / down_rate

    # Big_x = (cnt_Big[0, 0, 0] + cnt_Big[1, 0, 0] + cnt_Big[2, 0, 0] + cnt_Big[3, 0, 0]) / 4
    # Big_y = (cnt_Big[0, 0, 1] + cnt_Big[1, 0, 1] + cnt_Big[2, 0, 1] + cnt_Big[3, 0, 1]) / 4
    x_min = cnt_Big[0, 0, 0]
    y_min = cnt_Big[0, 0, 1]

    cnt[..., 0] = cnt[..., 0] - x_min
    cnt[..., 1] = cnt[..., 1] - y_min

    cnt[cnt < 0] = 0

    glom = img[int(cnt[0, 0, 1]):int(cnt[3, 0, 1]), int(cnt[0, 0, 0]):int(cnt[1, 0, 0])]

    return glom, cnt

def get_annotation_contour(img, contour, down_rate, shift, lv, start_x, start_y, end_x, end_y, resize_flag):
    vertices = contour['Vertices']['Vertex']
    cnt = np.zeros((4,1,2))

    now_id = int(contour['@Id'])

    cnt[0, 0, 0] = vertices[0]['@X']
    cnt[0, 0, 1] = vertices[0]['@Y']
    cnt[1, 0, 0] = vertices[1]['@X']
    cnt[1, 0, 1] = vertices[1]['@Y']
    cnt[2, 0, 0] = vertices[2]['@X']
    cnt[2, 0, 1] = vertices[2]['@Y']
    cnt[3, 0, 0] = vertices[3]['@X']
    cnt[3, 0, 1] = vertices[3]['@Y']

    cnt[0, 0, 0] = cnt[0, 0, 0] - shift
    cnt[1, 0, 0] = cnt[1, 0, 0] - shift
    cnt[2, 0, 0] = cnt[2, 0, 0] - shift
    cnt[3, 0, 0] = cnt[3, 0, 0] - shift

    cnt = cnt.astype(int)

    patch_size_x = int((cnt[2, 0, 0] - cnt[0, 0, 0]) / down_rate)
    patch_size_y = int((cnt[2, 0, 1] - cnt[0, 0, 1]) / down_rate)

    patch_start_x = cnt[0, 0, 0] + start_x
    patch_start_y = start_y + cnt[0, 0, 1]
    print(patch_start_x,patch_start_y,patch_size_x,patch_size_y)

    patch = np.array(img.read_region((patch_start_x, patch_start_y), lv, (patch_size_x, patch_size_y)).convert('RGB'))

    if resize_flag:
        patch_resize = resize(patch, (int(patch.shape[0]/ 2), int(patch.shape[1] / 2)))
        cnt = cnt / 2
    else:
        patch_resize = patch

    return patch_resize, cnt, now_id
def get_none_zero(black_arr):

    nonzeros = black_arr.nonzero()
    starting_y = nonzeros[0].min()
    ending_y = nonzeros[0].max()
    starting_x = nonzeros[1].min()
    ending_x = nonzeros[1].max()

    return starting_x, starting_y, ending_x, ending_y

def get_nonblack_starting_point(simg):
    px = 0
    py = 0
    black_img = simg.read_region((px, py), 3, (3000, 3000))
    starting_x, starting_y, ending_x, ending_y = get_none_zero(np.array(black_img)[:, :, 0])

    multiples = int(np.floor(simg.level_dimensions[0][0]/float(simg.level_dimensions[3][0])))

    #staring point
    px2 = (starting_x - 1) * multiples
    py2 = (starting_y - 1) * multiples
    #ending point
    px3 = (ending_x + 1) * multiples
    py3 = (ending_y + 1) * multiples

    # black_img_big = simg.read_region((px2, py2), 0, (1000, 1000))
    # offset_x, offset_y, offset_xx, offset_yy = get_none_zero(np.array(black_img_big)[:, :, 0])
    #
    # x = px2+offset_x
    # y = py2+offset_y

    xx, yy = scan_nonblack(simg, px2, py2, px3, py3)

    return xx,yy

def get_nonblack_ending_point(simg):
    px = 0
    py = 0
    black_img = simg.read_region((px, py), 3, (3000, 3000))
    starting_x, starting_y, ending_x, ending_y = get_none_zero(np.array(black_img)[:, :, 0])

    multiples = int(np.floor(simg.level_dimensions[0][0]/float(simg.level_dimensions[3][0])))

    #staring point
    px2 = (starting_x - 1) * multiples
    py2 = (starting_y - 1) * multiples
    #ending point
    px3 = (ending_x - 1) * (multiples-1)
    py3 = (ending_y - 1) * (multiples-1)

    # black_img_big = simg.read_region((px2, py2), 0, (1000, 1000))
    # offset_x, offset_y, offset_xx, offset_yy = get_none_zero(np.array(black_img_big)[:, :, 0])
    #
    # x = px2+offset_x
    # y = py2+offset_y

    xx, yy = scan_nonblack_end(simg, px2, py2, px3, py3)

    return xx,yy

def scan_nonblack(simg,px_start,py_start,px_end,py_end):
    offset_x = 0
    offset_y = 0
    line_x = py_end-py_start
    line_y = px_end-px_start

    val = simg.read_region((px_start+offset_x, py_start), 0, (1, 1))
    arr = np.array(val)[:, :, 0].sum()
    while arr == 0:
        val = simg.read_region((px_start+offset_x, py_start), 0, (1, line_x))
        arr = np.array(val)[:, :, 0].sum()
        offset_x = offset_x + 1

    val = simg.read_region((px_start, py_start+offset_y), 0, (1, 1))
    arr = np.array(val)[:, :, 0].sum()
    while arr == 0:
        val = simg.read_region((px_start, py_start+offset_y), 0, (line_y, 1))
        arr = np.array(val)[:, :, 0].sum()
        offset_y = offset_y + 1

    x = px_start+offset_x-1
    y = py_start+offset_y-1
    return x,y

def scan_nonblack_end(simg, px_start, py_start, px_end, py_end):
    offset_x = 0
    offset_y = 0
    line_x = py_end - py_start
    line_y = px_end - px_start

    val = simg.read_region((px_end + offset_x, py_end), 0, (1, 1))
    arr = np.array(val)[:, :, 0].sum()
    while not arr == 0:
        val = simg.read_region((px_end + offset_x, py_end), 0, (1, line_x))
        arr = np.array(val)[:, :, 0].sum()
        offset_x = offset_x + 1

    val = simg.read_region((px_end, py_end + offset_y), 0, (1, 1))
    arr = np.array(val)[:, :, 0].sum()
    while not arr == 0:
        val = simg.read_region((px_end, py_end + offset_y), 0, (line_y, 1))
        arr = np.array(val)[:, :, 0].sum()
        offset_y = offset_y + 1

    x = px_end + (offset_x - 1)
    y = py_end + (offset_y - 1)
    return x, y


directory = ['5X', '10X', '40X']


def preprocess_all():
    for dirpath, _, files in os.walk('svs'):
        if len(files) > 0:
            for dirname in directory:
                path = os.path.join(dirpath, dirname)
                os.makedirs(path, exist_ok=True)

    for dirpath, dirnames, files in os.walk('svs'):
        print(f'pre-processing directory: {dirpath}')
        for file_name in files:
            print(file_name)
            img = openslide.open_slide(dirpath + '/' + file_name)
            x, y = scan_nonblack_end(img, 0, 0, img.dimensions[0], img.dimensions[1])
            start = img.dimensions - np.array((x, y))

            filename_40X = dirpath + '/' + '40X/40X_' + file_name.replace('.svs', '.png')
            img.read_region(start, 0, img.dimensions).save(filename_40X)

            img_40X = plt.imread(filename_40X)

            img_10X = resize(img_40X, (int(img_40X.shape[0] / 4), int(img_40X.shape[1] / 4)))
            filename_10X = dirpath + '/' + '10X/10X_' + file_name.replace('.svs', '.png')
            plt.imsave(filename_10X, img_10X)

            img_5X = resize(img_40X, (int(img_40X.shape[0] / 8), int(img_40X.shape[1] / 8)))
            filename_5X = dirpath + '/' + '5X/5X_' + file_name.replace('.svs', '.png')
            plt.imsave(filename_5X, img_5X)


def preprocess_one(dirpath, filename):
    img = openslide.open_slide(dirpath + '/' + filename)
    x, y = scan_nonblack_end(img, 0, 0, img.dimensions[0], img.dimensions[1])
    start = img.dimensions - np.array((x, y))

    filename_40X = dirpath + '/' + '40X/40X_' + filename.replace('.svs', '.png')
    print('saving 40X png...')
    img.read_region(start, 0, img.dimensions).save(filename_40X)

    print('reading 40X png...')
    img_40X = plt.imread(filename_40X)

    print('resizing to 10X...')
    img_10X = resize(img_40X, (int(img_40X.shape[0] / 4), int(img_40X.shape[1] / 4)))
    filename_10X = dirpath + '/' + '10X/10X_' + filename.replace('.svs', '.png')
    print('saving 10X png...')
    del img_40X
    plt.imsave(filename_10X, img_10X)


    print('resizing to 5X...')
    img_5X = resize(img_10X, (int(img_10X.shape[0] / 2), int(img_10X.shape[1] / 2)))
    filename_5X = dirpath + '/' + '5X/5X_' + filename.replace('.svs', '.png')
    print('saving 5X png...')
    del img_10X
    plt.imsave(filename_5X, img_5X)

def find_properties(dirpath, filename):
    img = openslide.open_slide(dirpath + '/' + filename)
    print(img.properties)

def test(dirpath, filename):
    img = openslide.open_slide(dirpath + '/' + filename)
    print(img.properties)
    # print(img.level_dimensions)
    # x, y = scan_nonblack_end(img, 0, 0, img.dimensions[0], img.dimensions[1])
    # start = img.dimensions - np.array((x, y))
    # print(img.dimensions)
    # print(start)
    #
    # filename_40X = dirpath + '/' + '40X/40X_left' + filename.replace('.svs', '.png')
    # print('saving 40X png...')
    # img_left = img.read_region((0, 0), 0, (int(img.dimensions[0] / 2), img.dimensions[1]))
    # print(img_left.size)
    # print('saving left')
    # img_left.save(filename_40X)
    #
    # filename_40X = dirpath + '/' + '40X/40X_right' + filename.replace('.svs', '.png')
    # img_right = img.read_region((img.dimensions[0], 0), 0, (int(img.dimensions[0] / 2), img.dimensions[1]))
    # print(img_right.size)
    # print('saving right')
    # img_right.save(filename_40X)

    filename_40X = dirpath + '/' + '10X/10X_left' + filename.replace('.svs', '.png')
    # print(img.level_dimensions[1][0])
    # img_left = img.read_region((0, 0), 1, (int(img.level_dimensions[1][0] / 2), img.level_dimensions[1][1]))
    # print('saving...')def get_none_zero(black_arr):
    # img_left.save(filename_40X)

def read_test(dirpath, filename):
    filename_40X = dirpath + '/5X/0b8f60ca-2cb1-4e2b-83b8-a5ff6db96346_S-1909-007135_HE_2of2_1.png'
    img = plt.imread(filename_40X)
    print(img.shape)


def scn_to_png(svs_file,annotation_xml_file, output_folder, single_annotation):
    simg = openslide.open_slide(svs_file)
    print(simg.dimensions)
    name = os.path.basename(svs_file).replace('.svs', '')


    # read annotation region
    with open(annotation_xml_file) as fd:
        annotation_doc = xmltodict.parse(fd.read())
    annotation_layers = annotation_doc['Annotations']['Annotation']
    try:
        annotation_contours = annotation_layers['Regions']['Region']
    except:
        if len(annotation_layers) == 2:
            annotation_BBlayer = annotation_layers[0]
            annotation_regions = annotation_BBlayer['Regions']['Region']
            annotation_Masklayer = annotation_layers[1]
        else:
            annotation_Masklayer = annotation_layers[0]
        annotation_contours = annotation_Masklayer['Regions']['Region']


    # start_x, start_y = get_nonblack_starting_point(simg)
    end_x, end_y = 0, 0 #get_nonblack_ending_point(simg)
    #
    # print(start_x, start_y)
    # print(end_x,end_y)
    start_x, start_y = 0, 0
    if single_annotation:
        contour = annotation_contours
        patch_10X, cnt_10X, id = get_annotation_contour(simg, contour, 4, 0, 1, start_x, start_y, end_x, end_y, 0)
        patch_40X, cnt_40X, _ = get_annotation_contour(simg, contour, simg.level_downsamples[0], 0, 0, start_x, start_y,
                                                       end_x, end_y, 0)
        patch_5X, cnt_5X, _ = get_annotation_contour(simg, contour, 4, 0, 1, start_x, start_y, end_x, end_y, 1)

        X40_output_folder = os.path.join(output_folder, '40X')
        X5_output_folder = os.path.join(output_folder, '5X')
        X10_output_folder = os.path.join(output_folder, '10X')

        if not os.path.exists(X40_output_folder):
            os.makedirs(X40_output_folder)

        if not os.path.exists(X5_output_folder):
            os.makedirs(X5_output_folder)

        if not os.path.exists(X10_output_folder):
            os.makedirs(X10_output_folder)

        now_name = '%s_%s.png' % (name, id)
        plt.imsave(os.path.join(X40_output_folder, now_name), patch_40X)
        plt.imsave(os.path.join(X5_output_folder, now_name), patch_5X)
        plt.imsave(os.path.join(X10_output_folder, now_name), patch_10X)
    else:
        for ci in range(len(annotation_contours)):
            'get each boundary of slice and match the detection results'
            df_bigmap = pd.DataFrame(columns=['x', 'y', 't', 'l'])
            df_detection = pd.DataFrame(columns=['x', 'y', 't', 'l'])

            contour = annotation_contours[ci]
            patch_10X, cnt_10X, id = get_annotation_contour(simg, contour, 4, 0, 1, start_x, start_y, end_x, end_y, 0)
            patch_40X, cnt_40X, _ = get_annotation_contour(simg, contour, simg.level_downsamples[0], 0, 0, start_x, start_y, end_x, end_y, 0)
            patch_5X, cnt_5X, _ = get_annotation_contour(simg, contour, 4, 0, 1, start_x, start_y, end_x, end_y, 1)

            X40_output_folder = os.path.join(output_folder, '40X')
            X5_output_folder = os.path.join(output_folder,'5X')
            X10_output_folder = os.path.join(output_folder,'10X')

            if not os.path.exists(X40_output_folder):
                os.makedirs(X40_output_folder)

            if not os.path.exists(X5_output_folder):
                os.makedirs(X5_output_folder)

            if not os.path.exists(X10_output_folder):
                os.makedirs(X10_output_folder)

            now_name = '%s_%s.png' % (name, id)
            plt.imsave(os.path.join(X40_output_folder, now_name), patch_40X)
            plt.imsave(os.path.join(X5_output_folder, now_name), patch_5X)
            plt.imsave(os.path.join(X10_output_folder, now_name), patch_10X)


if __name__ == '__main__':
    dirpath = 'svs/PAS'
    filename = '3daf2299-1c81-4a33-9596-0acd09e340e7_S-1909-007149_PAS_1of2.svs'

    # annotation file
    now_annotation_xml = 'PAS_3daf.xml'

    # single_annotation indicates that whether the .xml file only contain single region of annotation.
    scn_to_png(dirpath + '/' + filename, dirpath + '/' + now_annotation_xml, dirpath, single_annotation=True)

