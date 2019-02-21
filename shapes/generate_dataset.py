from image_utils import get_image, SHAPE_CIRCLE, SHAPE_SQUARE, SHAPE_TRIANGLE, COLOR_RED, COLOR_GREEN, COLOR_BLUE
import numpy as np
from random import shuffle

def get_datasets(train_size, val_size, test_size, f_get_dataset, seed):
	train_data = f_get_dataset(train_size, seed)
	val_data = f_get_dataset(val_size, seed)
	test_data = f_get_dataset(test_size, seed)

	return train_data, val_data, test_data

def get_dataset_balanced(size, seed):
	np.random.seed(seed)

	data = []

	for i in range(size):
		data.append(get_image(seed+i))

	shuffle(data)

	return data


def get_dataset_unbalanced(size, seed, least_freq_shape=SHAPE_CIRCLE, least_freq_ratio=0.1):
	np.random.seed(seed)

	data = []

	n_unfreq_shapes = least_freq_ratio * size

	for i in range(size):
		if i < n_unfreq_shapes:
			shape = least_freq_shape
		else:
			shape = least_freq_shape + 1 if np.random.randint(2) == 0 else least_freq_shape + 2

		data.append(get_image(seed+i, shape))

	shuffle(data)

	return data
