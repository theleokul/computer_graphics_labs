#!/usr/bin/env python

import argparse

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
from skimage.io import imread
from skimage.transform import resize

from dataset import get


parser = argparse.ArgumentParser(description='Creates heat map for specified geoparams')
parser.add_argument('--lat', nargs=2, type=float, default=[-90.0, 90.0], help='Latitude range')
parser.add_argument('--lon', nargs=2, type=float, default=[0.0, 360.0], help='Longitude range')
parser.add_argument('-l', '--level', type=float, default=50.0, help='Depth level')
parser.add_argument('--name', type=str, default='color_map.png', help='Path for the output image')
args = parser.parse_args()

# Set geoparams from arguments
lat = tuple(args.lat)
lon = tuple(args.lon)
level = args.level
output_path = args.name


def resize_with_extender(ds, new_shape, extender):
    """Fill rest space of dataset with filler to be the same shape as new_shape"""
    prepared_ds = np.full((181, 361), extender)
    # import pdb; pdb.set_trace()
    prepared_ds[int(90 - lat[1]):int(91 - lat[0]), int(lon[0]):int(lon[1] + 1)] = ds
    prepared_ds = resize(prepared_ds, new_shape)
    return prepared_ds


def main():
    # Read world map
    world_map = imread('world_map2.jpg', as_gray=True)

    # Set waves dataset
    ds = get(lat, lon, level)
    ds = resize_with_extender(ds, world_map.shape, extender=ds.min())

    # Form color map
    plt.axis('off')
    plt.title('Color map', fontsize=20, pad=30)
    plt.imshow(world_map, cmap='gray')
    plt.imshow(ds, alpha=0.7, cmap='gist_heat')
    plt.colorbar(norm=Normalize(vmin=ds.min(), vmax=ds.max()), orientation='horizontal')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.2)

    print(f"Congratulations! The color map was created. Check out {output_path}.")


if __name__ == '__main__':
    main()
