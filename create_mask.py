import random
import os
import sys
import torch

import numpy as np
from matplotlib import cm, pyplot as plt
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon
from tqdm import tqdm

import rasterio as rs
import rasterio

from utils.utils import load_geotiff, pixel_to_coord, ray_tracing_numpy_numba
from utils.constants import CUTSIZEX, CUTSIZEY
import argparse

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--country', type=str, default="us", help="us, uganda, tanzania")
parser.add_argument('--district', type=str, default="new_york", help="new_york, north_dakota, tennessee, uganda")
parser.add_argument('--data_root', type=str, default="./data/sample_data")

parser.add_argument('--all_pixels', action='store_true')
parser.add_argument('--uniform_sampling', action='store_true')
parser.add_argument('--sampling_method', type=str, default="population", help="uniform, NL, population")
parser.add_argument('--save_data', action='store_true')
parser.add_argument('--load_data', action='store_true')
parser.add_argument('--overwrite', action='store_true')

# Run related parameters
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--batch_size', type=int, default=50000)

args = parser.parse_args()
device = "cpu"
args.device = device
country = args.country
sampling_method = args.sampling_method
district = args.district
district = district.lower()

seed = args.seed
data_root = args.data_root

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

NL_DATA = f"{args.data_root}/covariates/NL_raster.tif"
POP_DATA = f"{args.data_root}/covariates/population_raster.tif"

def get_index(name_list, district):
    for i in range(len(name_list)):
        name = name_list[i].lower()
        name = name.replace(" ", "_")
        if name == district:
            return i
    print("district {} not found in the us states".format(district))
    exit()

if __name__ == "__main__":
    if sampling_method in ['uniform', 'NL']:
        cutsizex = CUTSIZEX['NL'][country]
        cutsizey = CUTSIZEY['NL'][country]
    else:
        cutsizex = CUTSIZEX[sampling_method][country]
        cutsizey = CUTSIZEY[sampling_method][country]
    raster_data = f"{args.data_root}/covariates/{args.sampling_method}_raster.tif"
    
    file = f"{args.data_root}/{args.sampling_method}/{cutsizex[0]}_{cutsizex[1]}_{cutsizey[0]}_{cutsizey[1]}_{district}_mask.pth"
    if os.path.isfile(file):
        if not args.overwrite:
            exit()

    shapefile = gpd.read_file(
        os.path.join(f"{args.data_root}/shapefiles/us_states/cb_2018_us_state_20m.shp"))
    index = get_index(shapefile['NAME'], district)
    poly = shapefile['geometry'][index]

    print("Creating binary mask for district {}...".format(shapefile['NAME'][index]))
    channel = load_geotiff(raster_data)
    covariate_data = rs.open(raster_data)
    x_grid = np.meshgrid(np.arange(cutsizey[1] - cutsizey[0]), np.arange(cutsizex[1] - cutsizex[0]), sparse=False,
                         indexing='xy')  # faster
    grid = np.array(np.stack([x_grid[1].reshape(-1), x_grid[0].reshape(-1)], axis=1))
    cut = channel[cutsizex[0]:cutsizex[1], cutsizey[0]:cutsizey[1]]  # * (binary_m)
    probs = cut[grid[:, 0], grid[:, 1]] / cut.sum()
    gt_data_coordinate = pixel_to_coord(grid[:, 0], grid[:, 1], cutsizex[0], cutsizey[0],
                                        covariate_data.transform)

    points = gt_data_coordinate
    batch_size = args.batch_size
    results_arr = []
    assert len(points) % batch_size == 0, "Batch size should divide number of points."

    # poly = [poly] # might need to comment out if ploy is an array
    for t in tqdm(range(len(points) // batch_size)):
        results = np.zeros(len(points[t * batch_size:(t + 1) * batch_size]))
        if type(poly) == Polygon:
            temp_results = ray_tracing_numpy_numba(points[t * batch_size:(t + 1) * batch_size],
                                                   np.stack([poly.exterior.coords.xy[0], poly.exterior.coords.xy[1]],
                                                            axis=1))
            results += temp_results
        else:
            for i in (range(len(poly))):
                temp_results = ray_tracing_numpy_numba(points[t * batch_size:(t + 1) * batch_size], np.stack(
                    [poly[i].exterior.coords.xy[0], poly[i].exterior.coords.xy[1]], axis=1))
                results += temp_results

        results_arr.extend(results)

    results_arr = (np.array(results_arr) != 0)
    assert results_arr.sum() != 0, "Mask all zero."
    assert results_arr.sum() <= len(results_arr), "Too many points."

    binary_m = (results_arr).reshape(cutsizex[1]-cutsizex[0], cutsizey[1]-cutsizey[0])
    torch.save(binary_m, file)
    print("Binary mask created")