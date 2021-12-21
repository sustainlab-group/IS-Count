import numpy as np
from matplotlib import cm, pyplot as plt
import pandas as pd
import os
import csv
import torch
import random

import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon

import sys
from tqdm import trange
from tqdm import tqdm

from scipy import spatial

import rasterio as rs
from rasterio.plot import show
import rasterio

from osgeo import gdal

import geopy
import geopy.distance
import geoplot as gplt

import warnings
warnings.simplefilter('ignore')


# The code for ray tracing within a polygon https://stackoverflow.com/questions/36399381/whats-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
# more efficient (batch-wise approach)
def ray_tracing_numpy_numba(points, poly):
    x, y = points[:, 0], points[:, 1]
    n = len(poly)
    inside = np.zeros(len(x), np.bool_)
    p1x, p1y = poly[0]
    for i in range(n + 1):
        p2x, p2y = poly[i % n]
        idx = np.nonzero((y > min(p1y, p2y)) & (y <= max(p1y, p2y)) & (x <= max(p1x, p2x)))[0]
        if len(idx):  # <-- Fixed here. If idx is null skip comparisons below.
            if p1y != p2y:
                xints = (y[idx] - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x:
                inside[idx] = ~inside[idx]
            else:
                idxx = idx[x[idx] <= xints]
                inside[idxx] = ~inside[idxx]

        p1x, p1y = p2x, p2y
    return inside


# https://rasterio.readthedocs.io/en/latest/topics/georeferencing.html
# NOTE: A dataset’s pixel coordinate system has its origin at the “upper left”
def pixel_to_coord(x, y, shiftedx, shiftedy, transform):  # Upper-left corner point of the pixel
    x = x + shiftedx
    y = y + shiftedy
    concat = np.stack([y, x, np.ones_like(x)], axis=1)  # batch, 3
    concat = concat.reshape(*concat.shape, 1)  # batch, 3, 1
    matrix = np.array(transform).reshape(3, 3).reshape(1, 3, 3)  # batch, 3, 3
    x_y = np.matmul(matrix, concat)[:, :2, 0]
    # lon, lat
    return x_y 


def load_geotiff(file):
    ds = gdal.Open(file)
    image = np.array(ds.GetRasterBand(1).ReadAsArray())
    image = np.clip(image, a_min=0., a_max=1e20)
    return image


def coord_to_pixel(x, y, data_dir, shiftedx=0, shiftedy=0,
                   plot=False):  # Upper-left corner point of the pixel
    cut = load_geotiff(data_dir)
    covariate_data = rs.open(data_dir)
    concat = np.stack([x, y, np.ones_like(x)], axis=1)  # batch, 3
    concat = concat.reshape(*concat.shape, 1)  # batch, 3, 1
    matrix = np.array(covariate_data.transform).reshape(3, 3)
    matrix = np.linalg.inv(matrix)
    matrix = matrix.reshape(1, 3, 3)  # batch, 3, 3
    x_y = np.matmul(matrix, concat)[:, :2, 0]

    x = (x_y[:, 1] - shiftedx - 0.5).astype(int)  # round to the closest integer
    y = (x_y[:, 0] - shiftedy - 0.5).astype(int)  # round to the closest integer
    prob = cut[x, y] / cut.sum()
    if plot:
        plt.scatter(y, x, c=prob)
    return prob, np.stack([x, y], axis=1)


def coord_to_pixel_loaded(x, y, covariate_img, covariate_data, shiftedx=0, shiftedy=0,
                   plot=False):  # Upper-left corner point of the pixel
    cut = covariate_img
    concat = np.stack([x, y, np.ones_like(x)], axis=1)  # batch, 3
    concat = concat.reshape(*concat.shape, 1)  # batch, 3, 1
    matrix = np.array(covariate_data.transform).reshape(3, 3)
    matrix = np.linalg.inv(matrix)
    matrix = matrix.reshape(1, 3, 3)  # batch, 3, 3
    x_y = np.matmul(matrix, concat)[:, :2, 0]

    x = (x_y[:, 1] - shiftedx + 0.5).astype(int)  # round to the closest integer
    y = (x_y[:, 0] - shiftedy + 0.5).astype(int)  # round to the closest integer
    prob = cut[x, y] / cut.sum()
    if plot:
        plt.scatter(y, x, c=prob)
    return prob, np.stack([x, y], axis=1)


def create_data(image, uniform=False, all_pixels=False, N=5000, binary_m=None):
    grid = np.nonzero(binary_m)
    image = image[grid] # only unmasked pixels

    grid = np.array([
        (grid[0][i], grid[1][i]) for i in range(len(grid[0]))
    ])

    rotation_matrix = np.array([
        [0, -1],
        [1, 0]
    ])

    probs = image.reshape(-1) / sum(image.reshape(-1))
    if all_pixels:
        # ix = binary_m.reshape(-1)  # (probs != 0)
        pixels = grid
        probs = probs
        points = grid.astype(np.float32)
    else:
        if uniform:
            ix = np.random.choice(range(len(grid)), size=N, replace=True)
        else:
            ix = np.random.choice(range(len(grid)), size=N, replace=True, p=probs)

        pixels = grid[ix]
        probs = probs[ix]
        points = grid[ix].astype(np.float32)
    
    data = (points @ rotation_matrix).astype(np.float32)
    data[:, 1] += 1

    return data, pixels, probs


def get_bounding_box(lats, lons, size):
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    for lat, lon in zip(lats, lons):
        center = geopy.Point(lat, lon)
        diagonal_dist = (size * np.sqrt(2)) / 2
        d = geopy.distance.distance(kilometers=diagonal_dist)
        point0 = d.destination(point=center, bearing=225)  # Bottom-left
        point1 = d.destination(point=center, bearing=45)  # Top-right
        x0.append(point0[0])
        y0.append(point0[1])
        x1.append(point1[0])
        y1.append(point1[1])
    x0 = np.array(x0)
    y0 = np.array(y0)
    x1 = np.array(x1)
    y1 = np.array(y1)
    return [x0, x1, y0, y1]


def compute_pixel_size(lats, lons, raster_data_img, raster_data):
    _, pixels = coord_to_pixel_loaded(lons, lats, raster_data_img, raster_data, shiftedx=0, shiftedy=0, plot=False)
    raster_data_transform = raster_data.transform
    s_pix = []
    for i in trange(len(lons)):
        point0 = (lats[i], lons[i])
        point1 = np.array([pixels[i][0] + 1, pixels[i][1]])
        point2 = np.array([pixels[i][0], pixels[i][1] + 1])
        temp = np.stack([point1, point2], axis=1)
        temp = pixel_to_coord(temp[0], temp[1], 0, 0, raster_data_transform)
        point1 = np.roll(temp[0, :], 1)
        point2 = np.roll(temp[1, :], 1)
        x_diff = geopy.distance.distance(point0, point1).km
        y_diff = geopy.distance.distance(point0, point2).km
        area = np.abs(x_diff * y_diff)
        s_pix.append(area)
    s_pix = np.array(s_pix)
    return s_pix


def kd_tree_object_count(satellite_size, samples, sample_lats, sample_lons, tree, center_x, center_y, num_neighbor=5000):
    print("Querying tree...")
    neighbors = tree.query(samples, num_neighbor)  # find the k nearest neighbor
    dd, ii = np.array(neighbors)

    data_shape = ii.shape
    ii = ii.reshape(-1)
    ii = ii.astype(int)
    center_x, center_y = np.array(center_x), np.array(center_y)
    center_x_small = center_x[ii]
    center_y_small = center_y[ii]
    center_x_small = center_x_small.reshape(*data_shape)
    center_y_small = center_y_small.reshape(*data_shape)
    print("Getting bounding boxes...")
    lat0, lat1, lon0, lon1 = get_bounding_box(sample_lats, sample_lons, satellite_size)
    print("Creating masks")
    mask_lon = (center_x_small <= lon1.reshape(-1, 1)) * (center_x_small >= lon0.reshape(-1, 1))  # batch, num_neighbor
    mask_lat = (center_y_small <= lat1.reshape(-1, 1)) * (center_y_small >= lat0.reshape(-1, 1))  # batch, num_neighbor
    mask = mask_lat * mask_lon
    del (mask_lon, mask_lat)
    print("Masks created")
    object_count_array = np.sum(mask, axis=1)
    print("Object count collected")
    return object_count_array