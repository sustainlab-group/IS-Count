import argparse
import pdb
import random
import os
import sys
import math
import logging

import torch
import torch.optim as optim
import numpy as np

from matplotlib import cm, pyplot as plt
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm
from scipy import spatial

from utils.utils import kd_tree_object_count
from utils.constants import GT_MS_COUNT, GT_OPEN_BUILDINGS_COUNT, US_STATES, AFRICAN_COUNTRIES


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--country', type=str, default="us", help="us | uganda | tanzania | africa")
parser.add_argument('--district', type=str, default="new_york", help="new_york, north_dakota, tennessee, uganda")
parser.add_argument('--data_root', type=str, default="sample_data", help="root directory to data")

parser.add_argument('--sampling_method', type=str, default="NL", help="NL | population | uniform")
parser.add_argument('--save_data', action='store_true')
parser.add_argument('--load_data', action='store_true')
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--plot', action='store_true')

# Run related parameters
parser.add_argument('--num_run', type=int, default=5, help="number of estimation trials to run")
parser.add_argument('--group_run', type=int, default=1)

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--percentage', type=float, default=0.001, help="percentage of area covered by samples")

parser.add_argument('--total_sample_size', type=int, default=20000)
parser.add_argument('--trial_size', type=int, default=1000, help="number of samples used for estimation")
parser.add_argument('--satellite_size', type=float, default=640 * 0.0003, help="size of each sample tile (km)")

args = parser.parse_args()
device = torch.device('cuda:%d' % args.gpu)
args.device = device
country = args.country
district = args.district
seed = args.seed
data_root = args.data_root
sampling_method = args.sampling_method

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

if __name__ == "__main__":
    #####################
    ## Data Loading
    #####################
    output_txt = f"identity_{country}_{district}_{args.satellite_size}_{args.percentage}_{args.num_run}_{args.seed}.txt"
    os.makedirs(f"results/{args.sampling_method}", exist_ok=True)
    text_file = open(os.path.join(f"results/{args.sampling_method}", output_txt), "w")
    text_file.write("country district accuracy accuracy(std) error error(std) total_area(%) images est gt\n")

    if district == 'all':
        district_list = [country]
    else:
        district_list = [district]
        
    for district in district_list:
        text_file.write("{} {} ".format(country, district))
        print("{} {} ".format(country, district), flush=True)
        # Load ALL points in a district
        all_probs = {}

        if args.sampling_method != 'uniform':
            all_lats, all_lons, all_s_pix, all_probs['nl'], all_probs['pop'] = torch.load(f"{args.data_root}/{args.sampling_method}/sample_{country}_{district}_All_area.pth")
        else: # uniform sampling (using NL mask and pixel size)
            all_lats, all_lons, all_s_pix, all_probs['nl'], all_probs['pop'] = torch.load(f"{args.data_root}/NL/sample_{country}_{district}_All_area.pth")
        #####################
        ## Sampling
        #####################

        # Perform sampling
        print(f"Sampling from the {args.sampling_method} distribution...")

        if args.sampling_method == 'NL':
            probs = all_probs['nl'] * all_s_pix / (all_probs['nl'] * all_s_pix).sum()
            pred = all_probs['nl']
        elif args.sampling_method == 'population':
            probs = all_probs['pop'] * all_s_pix / (all_probs['pop'] * all_s_pix).sum()
            pred = all_probs['pop']
        else:
            area = all_s_pix.sum()
            probs = all_s_pix / all_s_pix.sum()
            pred = all_s_pix

        # Get necessary terms for estimating total count
        area = all_s_pix.sum()
        
        ##############################################
        # change total number of image to percentage
        ##############################################
        
        total_sample_size = area * args.percentage / (args.satellite_size ** 2)
        total_sample_size = int((total_sample_size // 20 + 1) * 20)
        args.trial_size = total_sample_size
        print("Sample size {}".format(args.trial_size))
        args.total_sample_size = args.trial_size * args.num_run

        ix = np.random.choice(range(len(all_lons)), size=args.total_sample_size, replace=True, p=probs)
        sample_lons, sample_lats, s_pix, pix_value, prob_subset = all_lons[ix], all_lats[ix], all_s_pix[ix], pred[ix], probs[ix]
        print("Sampling done...")
        
        # collect the correpsonding object counts
        object_count_array = []
        print("Collecting object count...")

        # Load MS ground truth building dataset
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

        points = np.stack([center_x, center_y], axis=1)
        samples = np.stack([sample_lons, sample_lats], axis=1)

        print("Building tree...")
        tree = spatial.KDTree(points)
        num_neighbor = 5000
        object_count_array = kd_tree_object_count(args.satellite_size, samples, sample_lats, sample_lons, tree, center_x, center_y, num_neighbor=num_neighbor)
        
        print("Computing accuracy...")
        accuracy_array = []
        for run in tqdm(range(args.num_run * args.group_run)):
            accuracy = []
            probs = prob_subset[run * args.trial_size: (run+1) * args.trial_size]
            assert (len(probs) == args.trial_size)
            object_count = object_count_array[run * args.trial_size : (run+1) * args.trial_size]
            pix_value_perm = pix_value[run * args.trial_size : (run+1) * args.trial_size]

            for sample_num in range(1, args.trial_size + 1, 20):
                s_image = args.satellite_size ** 2
                m = sample_num
                prob = pix_value_perm[:sample_num] / (all_s_pix * pred).sum()
                f_x = object_count[:sample_num]
                total_count = (1. / s_image) * (1.0 / prob) * f_x
                total_count = total_count.sum() / m
                accuracy.append(total_count / gt_count)
            accuracy_array.append(accuracy)
        accuracy_array = np.concatenate(accuracy_array, axis=0)
        accuracy_array = accuracy_array.reshape(args.num_run, args.group_run, -1).mean(axis=1)

        #########################
        ## Save evaluation plots
        #########################

        # Create accuracy plot
        mean = accuracy_array.mean(axis=0)
        std = accuracy_array.std(axis=0)
        
        print("Accuracy mean: ", mean[-1])
        print("Accuracy std: ", std[-1])

        text_file.write("{} {} ".format(mean[-1], std[-1]))
        logging.info(
            f"{country} accuracy {mean[-1]} {std[-1]}"
        )
        
        if args.plot:
            x_labels = range(1, args.trial_size + 1, 20)
            plt.plot(x_labels, mean, color="Tab:orange")
            plt.fill_between(x_labels, mean + std, mean - std, color="Tab:orange", alpha=0.3)
            plt.hlines(y=1.0, xmin=np.array(x_labels).min(), xmax=np.array(x_labels).max(), colors='tab:gray', linestyles=':')

            plt.ylabel("Accuracy", fontsize=20)
            plt.xlabel("Number of samples ({:.4f}%)".format(args.trial_size * 100 * s_image / area), fontsize=20)
            plt.ylim(0.2, 1.8)

            fig_name = "identity_accuracy_{}_{}_{}_{}_{}.png".format(country,
                                                                     district,
                                                                     args.num_run,
                                                                     args.group_run,
                                                                     args.trial_size,
                                                                     args.total_sample_size)
            plt.title("{} {}".format(country, args.sampling_method), fontsize=20)
            os.makedirs(f"figures/{sampling_method}/", exist_ok=True)
            plt.savefig(f"figures/{sampling_method}/{fig_name}")
            plt.close()

        # Create error plot
        error_array = np.abs(1. - accuracy_array)
        mean = error_array.mean(axis=0)
        std = error_array.std(axis=0)
        
        print("Error mean: ", mean[-1])
        print("Error std: ", std[-1])
        text_file.write("{} {} {} {} {} {}\n".format(mean[-1], std[-1], args.trial_size * 100 * s_image / area, args.trial_size, total_count, gt_count))
        text_file.flush()

        logging.info(
            f"{country} error {mean[-1]} {std[-1]}"
        )
        
        if args.plot:
            x_labels = range(1, args.trial_size + 1, 20)
            plt.plot(x_labels, mean, color="Tab:cyan")
            plt.fill_between(x_labels, mean + std, mean - std, color="Tab:cyan", alpha=0.3)

            plt.ylabel("Error Rate", fontsize=18)
            plt.xlabel("Number of samples ({:.4f}%)".format(args.trial_size * 100 * s_image / area), fontsize=18)
            plt.ylim(0., 1.)

            fig_name = "identity_error_{}_{}_{}_{}_{}.png".format(country,
                                                                  district,
                                                                  args.num_run,
                                                                  args.group_run,
                                                                  args.trial_size,
                                                                  args.total_sample_size)
            plt.title("{} {}".format(country, args.sampling_method), fontsize=20)
            plt.savefig(f"figures/{sampling_method}/{fig_name}")
    text_file.close()