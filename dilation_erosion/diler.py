#!/usr/bin/env python3

import sys
import argparse
import warnings

import numpy as np

from skimage.io import imread, imsave
from skimage import img_as_ubyte


if not sys.warnoptions:
    warnings.simplefilter("ignore")


def check_positive(val):
    ival = int(val)
    if ival <= 0:
        raise argparse.ArgumentTypeError(f'{val} is an invalid positive int value')
    return ival


def check_non_negative(val):
    ival = int(val)
    if ival < 0:
        raise argparse.ArgumentTypeError(f'{val} is an invalid positive int value')
    return ival


def guard(log_expr, error):
    if not log_expr:
        raise argparse.ArgumentError(error)


parser = argparse.ArgumentParser(description='Basic morphology operations: dilation, erosion')
parser.add_argument('-d', '--dilation', action='store_true', help='Dilation')
parser.add_argument('-e', '--erosion', action='store_true', help='Erosion')
parser.add_argument('-s', '--shape', type=str, choices=['rect', 'disk', 'ring'], help='Shape for structure element',
                    default='rect')
parser.add_argument('-m', '--measure', nargs=2, type=check_positive, help='Measure of the structure element',
                    required=True)
parser.add_argument('-o', '--origin', nargs=2, type=check_non_negative, help='Origin for structure element',
                    required=True)
parser.add_argument('-i', '--input', type=str, help='Path for input image', required=True)
args = parser.parse_args()

# Parse arguments
dilation, erosion = args.dilation, args.erosion
shape = args.shape
measure = args.measure
origin = args.origin
guard(origin[0] < measure[0] and origin[1] < measure[1], '-o,--origin out of structure element')
input_image_path = args.input


def disk(measure, filled=True):
    guard(measure[0] == measure[1] and measure[0] % 2 == 1, '-s,--shape should be odd and square')

    r = int(measure[0] / 2)
    x_c, y_c = r, r

    y, x = np.ogrid[-x_c:measure[0]-x_c, -y_c:measure[0]-y_c]
    mask = x*x + y*y <= r*r if filled else x*x + y*y == r*r

    disk = np.zeros(measure, dtype=np.uint8)
    disk[mask] = 255

    return disk


def structure_element(shape, measure, origin):
    strel = {'origin': origin}

    if shape == 'rect':
        strel['matrix'] = np.full(measure, 255, dtype=np.uint8)
    elif shape == 'disk':
        strel['matrix'] = disk(measure)
    elif shape == 'ring':
        strel['matrix'] = disk(measure, filled=False)
    else:
        raise argparse.ArgumentError('-s,--shape is unsupported')

    return strel


def discretized_colors(image):
    discr_image = np.copy(image)
    threshold = 100

    for ix, iy in np.ndindex(discr_image.shape):
        discr_image[ix, iy] = 0 if discr_image[ix, iy] <= threshold else 255

    return discr_image


def dilated_image(input_image, strel):
    dil_im = np.zeros(input_image.shape, dtype=np.uint8)
    origin = strel['origin']
    strel_matrix = strel['matrix']

    for ix, iy in np.ndindex(input_image.shape):
        if input_image[ix, iy] == 255:
            # Set x-bounds to change in dil_im
            _ix = ix - origin[0] if ix - origin[0] >= 0 else 0
            _iy = iy - origin[1] if iy - origin[1] >= 0 else 0

            # Set y-bounds to change in dil_im
            ix__, iy__ = ix + strel_matrix.shape[1] - origin[0], iy + strel_matrix.shape[0] - origin[1]
            ix_ = ix__ if ix__ < dil_im.shape[1] else dil_im.shape[1]
            iy_ = iy__ if iy__ < dil_im.shape[0] else dil_im.shape[0]

            # Perform union operation
            # import pdb; pdb.set_trace()
            dil_im[_ix:ix_, _iy:iy_] += strel_matrix[origin[0]-(ix-_ix):origin[0]+(ix_-ix),
                                                     origin[1]-(iy-_iy):origin[1]+(iy_-iy)]

            # Normalize data
            mask = dil_im[_ix:ix_, _iy:iy_].astype(bool)
            dil_im[_ix:ix_, _iy:iy_][mask] = 255

    return dil_im


def main():
    strel = structure_element(shape, measure, origin)
    input_image = discretized_colors(img_as_ubyte(imread(input_image_path, as_gray=True)))

    if not (dilation or erosion):
        # Do both dilation and erosion
        print('dilation and erosion')
    else:
        # Do specified morphology operations
        if dilation:
            # Do dilation
            dil_im = dilated_image(input_image, strel)
            imsave('dilated_image.png', dil_im)
        if erosion:
            # Do erosion
            print('erosion')


if __name__ == '__main__':
    main()
