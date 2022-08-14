
import glob
import timeit

import torch

import numpy as np
import cv2

import matplotlib.pyplot as plt


import os



def filter_contours(contours, hierarchy, filter_params):
    """
        Filter contours by: area.
    """
    filtered = []

    # find indices of foreground contours (parent == -1)
    hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)
    all_holes = []

    # loop through foreground contour indices
    for cont_idx in hierarchy_1:
        # actual contour
        cont = contours[cont_idx]
        # indices of holes contained in this contour (children of parent contour)
        holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
        # take contour area (includes holes)
        a = cv2.contourArea(cont)
        # calculate the contour area of each hole
        hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
        # actual area of foreground contour region
        a = a - np.array(hole_areas).sum()
        if a == 0: continue
        if tuple((filter_params['a_t'],)) < tuple((a,)):
            filtered.append(cont_idx)
            all_holes.append(holes)

    foreground_contours = [contours[cont_idx] for cont_idx in filtered]

    hole_contours = []

    for hole_ids in all_holes:
        unfiltered_holes = [contours[idx] for idx in hole_ids]
        unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
        # take max_n_holes largest holes by area
        unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
        filtered_holes = []

        # filter these holes
        for hole in unfilered_holes:
            if cv2.contourArea(hole) > filter_params['a_h']:
                filtered_holes.append(hole)

        hole_contours.append(filtered_holes)

    return foreground_contours, hole_contours


def getContour(big_slice_folder, output_dir, case_name):
    sections = glob.glob(os.path.join(big_slice_folder, '*' ))[0]

    img = (plt.imread(sections)[:, :, :3] * 255).astype(np.uint8)

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space

    mthresh = 59
    img_med = cv2.medianBlur(img_hsv[:, :, 1], mthresh)  # Apply median blurring

    # Thresholding
    # # if use_otsu:
    # _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # # else:

    sthresh = 20
    sthresh_up = 255

    _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

    # # Morphological closing
    # if close > 0:
    close = 4
    kernel = np.ones((close, close), np.uint8)
    img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)

    # Find and filter contours
    _, contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  # Find contours
    hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

    filter_params = {'a_t':20, 'a_h': 16, 'max_n_holes':8}
    foreground_contours, hole_contours = filter_contours(contours, hierarchy, filter_params)  # Necessary for filtering out artifacts

    contours_tissue = foreground_contours
    holes_tissue = hole_contours

    tissue_mask = get_seg_mask(region_size = img.shape, scale = 0, contours_tissue = contours_tissue, holes_tissue = holes_tissue, use_holes=True, offset=(0, 0))
    output_folder = os.path.join(output_dir.replace('clinical_patches', 'final_merge'), case_name)
    slice_merging_folder = os.path.join(output_folder.replace('clinical_patches', 'final_merge'))

    if not os.path.exists(slice_merging_folder):
        os.makedirs(slice_merging_folder)

    image_dir = os.path.join(slice_merging_folder, 'tissue_map_20X.npy')

    #plt.imsave(image_dir.replace('.npy','.png'), tissue_mask)

    np.save(image_dir, tissue_mask, allow_pickle=True)

    return tissue_mask[:, :, :1]


def get_seg_mask(region_size, scale, contours_tissue, holes_tissue, use_holes=False, offset=(0, 0)):
    print('\ncomputing foreground tissue mask')
    tissue_mask = np.full(region_size,0).astype(np.uint8)
    offset = tuple((np.array(offset) * np.array(scale) * -1).astype(np.int32))
    contours_holes = holes_tissue
    contours_tissue, contours_holes = zip(
        *sorted(zip(contours_tissue, contours_holes), key=lambda x: cv2.contourArea(x[0]), reverse=True))
    for idx in range(len(contours_tissue)):
        cv2.drawContours(image=tissue_mask, contours=contours_tissue, contourIdx=idx, color=(1,1,1), offset=offset,
                         thickness=-1)

        if use_holes:
            cv2.drawContours(image=tissue_mask, contours=contours_holes[idx], contourIdx=-1, color=(0,0,0),
                             offset=offset, thickness=-1)

    # # tissue_mask = tissue_mask.astype(bool)
    # print('detected {}/{} of region as tissue'.format(tissue_mask.sum(), tissue_mask.size))
    return tissue_mask.astype(np.float32)

if __name__ == "__main__":

    docker = 0
    gpu = 1

    start = timeit.default_timer()
    if gpu:
        if docker:
            print('using docker and gpu')
            # data_dir = '/desktop/src/extra/OmniSeg_MouthKidney_Pipeline/V11M25-279/20X'
            #data_dir = '/desktop/src/extra/OmniSeg_MouthKidney_Pipeline/svs_input/40X'
            data_dir = '/INPUTS/40X'
            # data_dir = '/Data2/HumanKidney/Mouse_Atubular_Segmentation/test'
            svs_folder = '/Data2/HumanKidney/2profile/circlenet/scn'
            output_dir = '/desktop/src/extra/OmniSeg_MouthKidney_Pipeline/clinical_patches'
            contour_dir = '/desktop/src/extra/OmniSeg_MouthKidney_Pipeline/final_merge'
        else:
            print('using local environment and gpu')
            # data_dir = '/desktop/src/extra/OmniSeg_MouthKidney_Pipeline/V11M25-279/20X'
            data_dir = '/home/lengh2/Desktop/Haoju_Leng/DockerFiles/test/src/extra/OmniSeg_MouthKidney_Pipeline/svs_input/40X'
            # data_dir = '/Data2/HumanKidney/Mouse_Atubular_Segmentation/test'
            svs_folder = '/Data2/HumanKidney/2profile/circlenet/scn'
            output_dir = '/home/lengh2/Desktop/Haoju_Leng/DockerFiles/test/src/extra/OmniSeg_MouthKidney_Pipeline/clinical_patches'
            contour_dir = '/home/lengh2/Desktop/Haoju_Leng/DockerFiles/test/src/extra/OmniSeg_MouthKidney_Pipeline/final_merge'

        sections = glob.glob(os.path.join(data_dir, '*'))
        sections.sort()

        for si in range(len(sections)):
            name = os.path.basename(sections[si])
            print(name)

            contour_map = getContour(data_dir, output_dir, name.replace('.png', ''))

            output_folder = sections[si].replace(data_dir, output_dir).replace('.png', '')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            img = plt.imread(sections[si])[:, :, :3]
            img = torch.from_numpy(img).cuda()
            threshold = img[0:2, 0:2, :3].mean()
            patch_size = 4096
            stride_size = 2048

            padding_size = patch_size - stride_size
# TODO: using loop
            img_padding = torch.ones(
                (img.shape[0] + 2 * padding_size, img.shape[1] + 2 * padding_size, 3)).cuda() * threshold

            img_padding[padding_size:-padding_size, padding_size:-padding_size, :] = img

            img = img_padding

            #plt.imsave('/home/lengh2/Desktop/Haoju_Leng/DockerFiles/test/src/extra/OmniSeg_MouthKidney_Pipeline/final_merge/padding.png',img.cpu().numpy())

            # contour_folder = name.replace('.png', '')
            # contour_dir = os.path.join(contour_dir, contour_folder)
            # image_dir = os.path.join(contour_dir, 'tissue_map_20X.npy')

            # contour_map = plt.imread(image_dir)[:, :, :1]
            #contour_map = np.load(image_dir, allow_pickle=True)[:, :, :1]
            contour_map[contour_map > 0.5] = 1
            contour_map[contour_map < 0.5] = 0

            contour_map = (torch.from_numpy(contour_map)).cuda().to(torch.uint8)

            contour_padding = torch.zeros(
                (contour_map.shape[0] + 2 * padding_size,contour_map.shape[1] + 2 * padding_size, 1)).cuda().to(torch.uint8)

            contour_padding[padding_size:-padding_size, padding_size:-padding_size, :] = contour_map

            contour_map = contour_padding

            stride_x = int(img.shape[0] / stride_size) - 1
            stride_y = int(img.shape[1] / stride_size) - 1

            for xi in range(stride_x):
                for yi in range(stride_y):

                    x_ind = int(xi * stride_size)
                    y_ind = int(yi * stride_size)

                    contour_patch = contour_map[x_ind:x_ind + patch_size, y_ind:y_ind + patch_size, :]
                    if 1 in contour_patch:
                        now_patch = img[x_ind:x_ind + patch_size, y_ind:y_ind + patch_size, :]
                        patch_dir = os.path.join(output_folder, '%d_%d.npy' % (x_ind, y_ind))
                    # if now_patch.mean() < threshold:
                        now_patch = now_patch.cpu().numpy()
                        # case_name = str(x_ind) + '_' + str(y_ind)
                        np.save(patch_dir, now_patch, allow_pickle=True)
                        #plt.imsave(patch_dir.replace('.npy', '.png'), now_patch)

        end = timeit.default_timer()
        print('step 1 duration:', end - start, 'seconds')


    else:
        if docker:
            print('using docker and cpu')
            # data_dir = '/desktop/src/extra/OmniSeg_MouthKidney_Pipeline/V11M25-279/20X'
            data_dir = '/desktop/src/extra/OmniSeg_MouthKidney_Pipeline/svs_input/40X'
            # data_dir = '/Data2/HumanKidney/Mouse_Atubular_Segmentation/test'
            svs_folder = '/Data2/HumanKidney/2profile/circlenet/scn'
            output_dir = '/desktop/src/extra/OmniSeg_MouthKidney_Pipeline/clinical_patches'
        else:
            print('using local environment and cpu')
            # data_dir = '/desktop/src/extra/OmniSeg_MouthKidney_Pipeline/V11M25-279/20X'
            data_dir = '/home/lengh2/Desktop/Haoju_Leng/DockerFiles/test/src/extra/OmniSeg_MouthKidney_Pipeline/svs_input/40X'
            # data_dir = '/Data2/HumanKidney/Mouse_Atubular_Segmentation/test'
            svs_folder = '/Data2/HumanKidney/2profile/circlenet/scn'
            output_dir = '/home/lengh2/Desktop/Haoju_Leng/DockerFiles/test/src/extra/OmniSeg_MouthKidney_Pipeline/clinical_patches'

        sections = glob.glob(os.path.join(data_dir, '*'))
        sections.sort()

        for si in range(len(sections)):
            name = os.path.basename(sections[si])
            print(name)

            output_folder = sections[si].replace(data_dir, output_dir).replace('.png', '')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            img = plt.imread(sections[si])[:,:,:3]


            patch_size = 2048
            stride_size = 1024

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
                    patch_dir = os.path.join(output_folder, '%d_%d.npy' % (x_ind, y_ind))

                    if now_patch.mean() < 220. / 255:
                        #plt.imsave(patch_dir, now_patch)
                        np.save(patch_dir, now_patch, allow_pickle=True)

        end = timeit.default_timer()
        print('step 1 duration:', end - start, 'seconds')

