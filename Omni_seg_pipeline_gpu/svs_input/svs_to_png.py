import openslide
import numpy as np
from skimage.transform import rescale, resize
import skimage.io
import matplotlib.pyplot as plt

img = openslide.open_slide('01d62c0b-4776-48c4-8445-6ffe6ee7ba88_S-1904-008114_HE_1of2.svs')


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


# print(img.properties)

# img.read_region((1,1), 0, img.dimensions).save('test.png')

img = plt.imread('test.png')
img_10X = resize(img, (int(img.shape[0] / 4), int(img.shape[1] / 4)))
plt.imsave('10X.png', img_10X)

img_5X = resize(img, (int(img.shape[0] / 8), int(img.shape[1] / 8)))
print(img_5X.shape, 'hhh')
plt.imsave('5X.png', img_5X)

img = plt.imread('10X.png')
print(img.shape)





