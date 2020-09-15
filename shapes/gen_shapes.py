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
cmd_parser.add_argument('--dataset_type', type=int, default=dataset_type)
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

if __name__ == "__main__":

    debugging = False

    if debugging:
        print('=============== Debugging ===================================')

    folder_name = 'shapes/{}'.format(shapes_dataset)
    #get_dataset_different_targets_zero_shot
    #get_dataset_balanced_zero_shot
    #get_dataset_uneven_incomplete#get_dataset_different_targets_incomplete#get_dataset_uneven_different_targets #get_dataset_balanced_incomplete #get_dataset_uneven #get_dataset_different_targets_three_figures#get_dataset_different_targets

    # seed = 42
    # np.random.seed(seed)

    # From Serhii's original experiment
    # train_size = 74504 if not debugging else 10
    # val_size = 8279 if not debugging else 10
    # test_size = 40504 if not debugging else 10

    train_size = 7450 if not debugging else 10
    val_size = 827 if not debugging else 10
    test_size = 4050 if not debugging else 10

    is_uneven = (f_generate_dataset is get_dataset_uneven
                or f_generate_dataset is get_dataset_uneven_different_targets
                or f_generate_dataset is get_dataset_uneven_incomplete
                or f_generate_dataset is get_dataset_uneven_different_targets_row
                or f_generate_dataset is get_dataset_uneven_different_targets_size
                or f_generate_dataset is get_dataset_uneven_different_targets_row_incomplete)
    if is_uneven:
        train_data, val_data, test_data, shapes_probs, colors_probs = get_datasets(train_size, val_size, test_size, f_generate_dataset, is_uneven)
    else:
        train_data, val_data, test_data = get_datasets(train_size, val_size, test_size, f_generate_dataset, is_uneven)

    has_tuples = type(train_data[0]) is tuple


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

    assert not os.path.exists(folder_name), 'Trying to overwrite?'
    os.mkdir(folder_name)

    for set_name, set_data in sets.items():
        if not has_tuples:
            set_inputs = np.asarray([image.data[:,:,0:3] for image in set_data])
        else:
            tuple_len = len(set_data[0]) # 2
            n_rows = len(set_data)
            set_inputs = np.zeros((n_rows, tuple_len, WIDTH, HEIGHT, N_CHANNELS), dtype=np.uint8)
            for i in range(n_rows):
                for j in range(tuple_len):
                    set_inputs[i][j] = set_data[i][j].data[:,:,0:3]

        random = np.random.normal(0, 1, size=set_inputs.shape)
        random[random>1] = 1
        random[random<-1] = -1
        random = np.array((random + 1)/2*noise_strength, dtype=np.uint8)
        set_inputs = np.array(set_inputs, dtype = np.uint16)
        set_inputs = set_inputs + random
        set_inputs[set_inputs>255] = 255
        set_inputs = np.array(set_inputs, dtype = np.uint8)

        np.save("{}/{}.input".format(folder_name, set_name), set_inputs)

        if not has_tuples:
            set_metadata = [image.metadata for image in set_data]
        else:
            set_metadata = [(image[0].metadata, image[1].metadata) for image in set_data]

        pickle.dump(set_metadata, open('{}/{}.metadata.p'.format(folder_name, set_name), 'wb'))

        if set_name=='val':
            random = np.random.normal(0, 1, size=set_inputs.shape)
            random[random>1] = 1
            random[random<-1] = -1
            noise_inputs = np.array((random + 1)/2*255, dtype=np.uint8)

            np.save("{}/{}.input".format(folder_name, 'noise'), noise_inputs)

            pickle.dump(set_metadata, open('{}/{}.metadata.p'.format(folder_name, 'noise'), 'wb'))

    if is_uneven:
        with open('{}/probs.txt'.format(folder_name), 'w') as f:
            f.write("P(circle)={}\n".format(shapes_probs[0]))
            f.write("P(square)={}\n".format(shapes_probs[1]))
            f.write("P(triangle)={}\n".format(shapes_probs[2]))
            f.write("P(red|circle)={}\n".format(colors_probs[0]))
            f.write("P(green|circle)={}\n".format(colors_probs[1]))
            f.write("P(blue|circle)={}\n".format(colors_probs[2]))
            f.write("P(red|square)={}\n".format(colors_probs[3]))
            f.write("P(green|square)={}\n".format(colors_probs[4]))
            f.write("P(blue|square)={}\n".format(colors_probs[5]))
            f.write("P(red|triangle)={}\n".format(colors_probs[6]))
            f.write("P(green|triangle)={}\n".format(colors_probs[7]))
            f.write("P(blue|triangle)={}\n".format(colors_probs[8]))

    print('Saved data set to folder {}'.format(folder_name))
