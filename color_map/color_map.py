import argparse

import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread
from skimage.color import gray2rgb
from skimage import img_as_float

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


def get_diff(im1, im2):
    """Creates an image based on difference between im1 and im2"""
    # Calculate the absolute difference on each channel separately
    error_r = np.fabs(np.subtract(im1[:, :, 0], im2[:, :, 0]))
    error_g = np.fabs(np.subtract(im1[:, :, 1], im2[:, :, 1]))
    error_b = np.fabs(np.subtract(im1[:, :, 2], im2[:, :, 2]))

    # Create diff image
    diff_img = np.maximum(np.maximum(error_r, error_g), error_b)

    return diff_img


def combine_and_save(*images, filename='color_map.jpg'):
    """Combines images and saves them to specified file"""
    fig, ax = plt.subplots()
    ax.grid(False)
    ax.axis('off')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    for i, image in enumerate(images, 1):
        fig.add_subplot(1, len(images), i)
        plt.imshow(image)
        plt.axis('off')

    plt.colorbar()
    plt.savefig(filename)


def main():
    world_map = imread('world_map.jpg')
    world_map = img_as_float(world_map)

    ds = get_dataset(lat, lon, level)
    ds = fit_ds(ds)
    ds = prepare_ds_shape(ds, world_map.shape[:-1])
    ds = gray2rgb(ds)

    diff_map = get_diff(world_map, ds)
    blended_map = 0.9 * ds + 0.1 * world_map

    combine_and_save(blended_map, diff_map)


if __name__ == '__main__':
    main()
