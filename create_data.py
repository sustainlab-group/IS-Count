import random
import os
import sys
import torch
import logging

import numpy as np
from matplotlib import cm, pyplot as plt
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon
from tqdm import tqdm

import rasterio as rs
import rasterio

from utils.utils import load_geotiff, create_data, pixel_to_coord, coord_to_pixel_loaded, compute_pixel_size, kd_tree_object_count
from utils.constants import US_STATES, AFRICAN_COUNTRIES, CUTSIZEX, CUTSIZEY, GT_MS_COUNT, GT_OPEN_BUILDINGS_COUNT
import argparse

from scipy import spatial

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--country', type=str, default="us", help="us, uganda, tanzania, africa")
parser.add_argument('--district', type=str, default="all", help="new_york, north_dakota, tennessee, uganda")
parser.add_argument('--data_root', type=str, default="./data/sample_data")
parser.add_argument('--all_pixels', action='store_true')
parser.add_argument('--sampling_method', type=str, default="NL", help="NL, population")
parser.add_argument('--overwrite', action='store_true')

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--total_sample_size', type=int, default=2000)
parser.add_argument('--satellite_size', type=float, default=640 * 0.0003)


args = parser.parse_args()
device = "cpu"
args.device = device
country = args.country
district = args.district
sampling_method = args.sampling_method
seed = args.seed

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


if __name__ == "__main__":
    # Directories to the covariate data
    nl_data = f"{args.data_root}/covariates/NL_raster.tif"
    pop_data = f"{args.data_root}/covariates/population_raster.tif"

    print("Loading covariate data...")
    raster_nl = rs.open(nl_data)
    raster_nl_img = load_geotiff(nl_data)
    raster_pop = rs.open(pop_data)
    raster_pop_img = load_geotiff(pop_data)
    print("Data loaded")


    # Load the base raster to conduct uniform sampling on
    if sampling_method == 'population':
        raster_data = raster_pop
        raster_data_img = raster_pop_img
    elif sampling_method == 'NL':
        raster_data = raster_nl
        raster_data_img = raster_nl_img
    else:
        raise NotImplementedError

    if district == 'all':
        district_list = [country]
    else:
        district_list = [district]
    
    for district in district_list:
        try:
            file = f'{args.data_root}/{sampling_method}/sample_{country}_{district}_All_area.pth'
            if os.path.isfile(file) and (not args.overwrite):
                continue

            logging.info(f"processing {country} {district}")
            print(f"processing {country} {district}", flush=True)
        
            if country in ['us', 'bangladesh']:
                cutsizex = CUTSIZEX[sampling_method][country]
                cutsizey = CUTSIZEY[sampling_method][country]
            else:
                cutsizex = CUTSIZEX[sampling_method][district]
                cutsizey = CUTSIZEY[sampling_method][district]

            print("Country {}, district {}".format(country, district))
            pth_mask = f'{args.data_root}/{sampling_method}/{cutsizex[0]}_{cutsizex[1]}_{cutsizey[0]}_{cutsizey[1]}_{district}_mask.pth'
            if not os.path.isfile(pth_mask):
                print("mask {} not exist {} {}".format(pth_mask, country, district), flush=True)
                continue

            binary_m = torch.load(f'{args.data_root}/{sampling_method}/{cutsizex[0]}_{cutsizex[1]}_{cutsizey[0]}_{cutsizey[1]}_{district}_mask.pth')
            cut = binary_m
            print(binary_m.sum())
            
            # Load ground truth building dataset
            if country == 'us':
                if district in US_STATES:
                    gt_count = GT_MS_COUNT[district]
                elif district == 'all':
                    gt_count = GT_MS_COUNT[country]
                [center_x, center_y] = torch.load(f"{args.data_root}/ms_building_footprint/us/{''.join(district.split('_'))}_center.pth")
                center_x, center_y = np.array(center_x), np.array(center_y)
            elif country == 'bangladesh':
                data_csv = pd.read_csv(f"{args.data_root}/brick_data/all_pos_without_shape_coords.csv")
                center_x = np.array(data_csv['long'])
                center_y = np.array(data_csv['lat'])
            else:
                [center_x, center_y] = torch.load(f"{args.data_root}/open_buildings/{district}_center.pth")
                center_x, center_y = np.array(center_x), np.array(center_y)
            
            #####################
            ## Positive samples
            #####################
            
            print('Creating positive data...')
            np.random.seed(args.seed)
            ix = np.random.choice(range(len(center_x)), size=args.total_sample_size, replace=False)
            pos_lons = np.array(center_x[ix])
            pos_lats = np.array(center_y[ix])

            print('Collecting object count...')
            points = np.stack([center_x, center_y], axis=1)
            samples = np.stack([pos_lons, pos_lats], axis=1)

            print("Building tree...")
            tree = spatial.KDTree(points)
            print("done")
            num_neighbor = 5000
            object_count_array = kd_tree_object_count(args.satellite_size, samples, pos_lats, pos_lons, tree, center_x, center_y, num_neighbor=num_neighbor)
            print('Object count collected')

            probs_nl, _ = coord_to_pixel_loaded(pos_lons, pos_lats, raster_nl_img, raster_nl, shiftedx=0, shiftedy=0, plot=False)
            probs_pop, _ = coord_to_pixel_loaded(pos_lons, pos_lats, raster_pop_img, raster_pop, shiftedx=0, shiftedy=0, plot=False)

            os.makedirs(f'{args.data_root}/{sampling_method}/', exist_ok=True)
            file = f'{args.data_root}/{sampling_method}/sample_{args.total_sample_size}_{country}_{district}_True.pth'
            if not os.path.isfile(file) or args.overwrite:
                torch.save([pos_lats, pos_lons, probs_nl, probs_pop, object_count_array], file)
            del(object_count_array)
            print('Positive data created')

            #####################
            ## Negative samples
            #####################
            
            print('Creating negative data...')
            _, pixels, _ = create_data(cut, all_pixels=False,
                                       uniform=True,
                                       N=args.total_sample_size,
                                       binary_m=binary_m)
            data_coordinate = pixel_to_coord(pixels[:, 0], pixels[:, 1], cutsizex[0], cutsizey[0],
                                             raster_data.transform)
            neg_lons = data_coordinate[:, 0]
            neg_lats = data_coordinate[:, 1]

            print('Collecting object count...')
            samples = np.stack([neg_lons, neg_lats], axis=1)

            num_neighbor = 5000
            object_count_array = kd_tree_object_count(args.satellite_size, samples, neg_lats, neg_lons, tree, center_x, center_y,
                                                      num_neighbor=num_neighbor)

            probs_nl, _ = coord_to_pixel_loaded(neg_lons, neg_lats, raster_nl_img, raster_nl, shiftedx=0, shiftedy=0, plot=False)
            probs_pop, _ = coord_to_pixel_loaded(neg_lons, neg_lats, raster_pop_img, raster_pop, shiftedx=0, shiftedy=0, plot=False)

            os.makedirs(f'{args.data_root}/{sampling_method}/', exist_ok=True)
            file = f'{args.data_root}/{sampling_method}/sample_{args.total_sample_size}_{country}_{district}_False.pth'
            if not os.path.isfile(file) or args.overwrite:
                torch.save([neg_lats, neg_lons, probs_nl, probs_pop, object_count_array], file)
            del(object_count_array)
            print('Negative data created')

            #####################
            ## All test samples
            #####################

            print('Creating all test data...')
            _, pixels, _ = create_data(cut, all_pixels=True,
                                          uniform=True,
                                          N=20000,
                                          binary_m=binary_m)
            data_coordinate = pixel_to_coord(pixels[:, 0], pixels[:, 1], cutsizex[0], cutsizey[0],
                                             raster_data.transform)
            lons = data_coordinate[:, 0]
            lats = data_coordinate[:, 1]

            probs_nl, _ = coord_to_pixel_loaded(lons, lats, raster_nl_img, raster_nl, shiftedx=0, shiftedy=0, plot=False)
            probs_pop, _ = coord_to_pixel_loaded(lons, lats, raster_pop_img, raster_pop, shiftedx=0, shiftedy=0, plot=False)
            
            print('Collecting pixel sizes...')
            s_pix = compute_pixel_size(lats, lons, raster_data_img, raster_data)
            print('Pixel sizes collected')

            os.makedirs(f'{args.data_root}/{sampling_method}/', exist_ok=True)
            file = f'{args.data_root}/{sampling_method}/sample_{country}_{district}_All_area.pth'
            if not os.path.isfile(file) or args.overwrite:
                torch.save([lats, lons, s_pix, probs_nl, probs_pop], file)

            print('Test data created')

        except:
            logging.info(f"ERROR {country} {district}")
            print(f"ERROR {country} {district}\n", flush=True)