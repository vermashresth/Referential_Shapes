#!/usr/bin/env python2

import numpy as np
import os

from generate_dataset import get_datasets, get_dataset_unbalanced

N_TRAIN_TINY    = 1
N_TRAIN_SMALL = 10
N_TRAIN_MED     = 100
N_TRAIN_LARGE = 1000
N_TRAIN_ALL     = N_TRAIN_MED



if __name__ == "__main__":

    folder_name = 'normal_dist'
    k = 3

    # From Serhii's original experiment
    train_size = 18626
    val_size = 2069
    test_size = 10126

    train_data, val_data, test_data = get_datasets(train_size, val_size, test_size, get_dataset_unbalanced)

    train_data_tiny = train_data[:N_TRAIN_TINY]
    train_data_small = train_data[:N_TRAIN_SMALL]
    train_data_med = train_data[:N_TRAIN_MED]
    train_data_large = train_data

    sets = {
        "train.tiny": train_data_tiny,
        "train.small": train_data_small,
        "train.med": train_data_med,
        "train.large": train_data_large,
        "val": val_data,
        "test": test_data
    }

    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

    for set_name, set_data in sets.items():
        set_inputs = np.asarray([image.data[:,:,0:3] for image in set_data])
        np.save("{}/{}.input".format(folder_name, set_name), set_inputs)
