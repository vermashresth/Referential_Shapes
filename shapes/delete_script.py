#!/usr/bin/env python2

import numpy as np
import os
import pickle



import argparse

N_TRAIN_TINY    = 1
N_TRAIN_SMALL = 10
N_TRAIN_MED     = 100
N_TRAIN_LARGE = 1000
N_TRAIN_ALL     = N_TRAIN_MED

noise_strength = 0
dataset_type = 0
use_bullet = 0
cmd_parser = argparse.ArgumentParser()
cmd_parser.add_argument('--noise_strength', type=int, default=noise_strength)
cmd_parser.add_argument('--dataset_type', type=int)
cmd_parser.add_argument('--use_bullet', type=int, default=use_bullet)

cmd_args = cmd_parser.parse_args()

noise_strength = cmd_args.noise_strength
dataset_type = cmd_args.dataset_type
use_bullet = cmd_args.use_bullet
if not use_bullet:
    from generate_dataset import *
    from image_utils import *
else:
    from generate_dataset import *
    activate_bullet()
    from bullet_image_utils import *

if dataset_type == 0: # Even, same pos
    shapes_dataset = 'get_dataset_balanced_incomplete_noise_{}_{}_{}'.format(noise_strength, N_CELLS, N_CELLS)
    dataset_name = 'even-samepos'
    f_generate_dataset = get_dataset_balanced_incomplete
elif dataset_type == 1: # Even, diff pos
    shapes_dataset = 'get_dataset_different_targets_incomplete_noise_{}_{}_{}'.format(noise_strength, N_CELLS, N_CELLS)
    dataset_name = 'even-diffpos'
    f_generate_dataset = get_dataset_different_targets_incomplete
elif dataset_type == 2: # Uneven, same pos
    shapes_dataset = 'get_dataset_uneven_incomplete_noise_{}_{}_{}'.format(noise_strength, N_CELLS, N_CELLS)
    dataset_name = 'uneven-samepos'
    f_generate_dataset = get_dataset_uneven_incomplete
elif dataset_type == 3: # Uneven,  diff pos
    shapes_dataset = 'get_dataset_uneven_different_targets_row_incomplete_noise_{}_{}_{}'.format(noise_strength, N_CELLS, N_CELLS)
    dataset_name = 'uneven-diffpos'
    f_generate_dataset = get_dataset_uneven_different_targets_row_incomplete
elif dataset_type == 4: #
    shapes_dataset = 'get_dataset_balanced_zero_shot_noise_{}_{}_{}'.format(noise_strength, N_CELLS, N_CELLS)
    dataset_name = 'zero-shot'
    f_generate_dataset = get_dataset_balanced_zero_shot
else:
    print("Not Supported type")

folder_name = 'shapes/{}'.format(shapes_dataset)
import shutil
shutil.rmtree(folder_name)
