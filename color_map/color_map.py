#!/usr/bin/env python

import argparse

import numpy as np
from skimage.io import imread, imsave
from skimage.color import gray2rgb, rgb2gray
from skimage import img_as_float
from skimage.transform import rotate

import imageio.core.util

from mine_dataset import get_dataset


parser = argparse.ArgumentParser(description='Creates heat map for specified geoparams')
parser.add_argument('--lat', nargs=2, type=float, default=[-90.0, 89.0], help='Latitude range')
parser.add_argument('--lon', nargs=2, type=float, default=[0.0, 359.0], help='Longitude range')
parser.add_argument('-l', '--level', type=float, default=50.0, help='Depth level')
args = parser.parse_args()

# Set geoparams from arguments
lat = tuple(args.lat)
lon = tuple(args.lon)
level = args.level


def ignore_warnings(*args, **kwargs):
    pass


def prepare_ds_shape(ds, world_map_shape):
    """Fill rest space of dataset with zeros to be the same shape as world_map"""
    prepared_ds = np.zeros(world_map_shape, dtype=float)
    prepared_ds[int(lon[0]):int(lon[1] + 1), int(lat[0] + 90.0):int(lat[1] + 91.0)] = ds
    return prepared_ds


def fit_ds(ds):
    """Fit dataset's content to standart float format (all values between 0.0 and 1.0)"""
    min = ds.min()
    if min < 0:
        ds = ds + abs(min)

    max = ds.max()
    if max > 1:
        ds = ds / abs(max)

    return ds


def redden_ds(ds):
    return ds * [1, 0, 0]


def main():
    # Dirty hack
    imageio.core.util._precision_warn = ignore_warnings

    # Read world map
    world_map = imread('world_map.jpg')
    world_map = img_as_float(world_map)
    world_map = rotate(world_map, angle=90, resize=True)
    world_map = gray2rgb(rgb2gray(world_map))

    # Set waves dataset
    ds = get_dataset(lat, lon, level)
    ds = fit_ds(ds)
    ds = prepare_ds_shape(ds, world_map.shape[:-1])
    ds = gray2rgb(ds)
    ds = redden_ds(ds)

    # Create output
    blended_map = 0.6 * ds + 0.4 * world_map
    blended_map = rotate(blended_map, angle=-90, resize=True)

    # Save output
    imsave('color_map.jpg', blended_map)

    print('Congratulations! Color map was created. Check out color_map.jpg')


if __name__ == '__main__':
    main()
