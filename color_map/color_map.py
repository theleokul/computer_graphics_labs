#!/usr/bin/env python

import argparse

import numpy as np
from skimage.io import imread, imsave
from skimage.color import gray2rgb
from skimage import img_as_float

import imageio.core.util

from mine_dataset import get_ds


parser = argparse.ArgumentParser(description='Creates heat map for specified geoparams')
parser.add_argument('--lat', nargs=2, type=float, default=[-90.0, 89.0], help='Latitude range')
parser.add_argument('--lon', nargs=2, type=float, default=[0.0, 359.0], help='Longitude range')
parser.add_argument('-l', '--level', type=float, default=50.0, help='Depth level')
parser.add_argument('--name', type=str, default='color_map.jpg', help='Path for the output image')
args = parser.parse_args()

# Set geoparams from arguments
lat = tuple(args.lat)
lon = tuple(args.lon)
level = args.level
output_path = args.name


def ignore_warnings(*args, **kwargs):
    pass


def prepare_ds_shape(ds, world_map_shape):
    """Fill rest space of dataset with zeros to be the same shape as world_map"""
    prepared_ds = np.zeros(world_map_shape, dtype=float)
    prepared_ds[int(lat[0] + 90.0):int(lat[1] + 91.0), int(lon[0]):int(lon[1] + 1)] = ds
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
    world_map = gray2rgb(world_map)

    # Set waves dataset
    ds = get_ds(lat, lon, level)
    ds = fit_ds(ds)
    ds = prepare_ds_shape(ds, world_map.shape[:-1])
    ds = gray2rgb(ds)
    ds = redden_ds(ds)

    # Create output
    blended_map = 0.6 * ds + 0.4 * world_map

    # Save output
    imsave(output_path, blended_map)

    print(f'Congratulations! The color map was created. Check out {output_path}.')


if __name__ == '__main__':
    main()
