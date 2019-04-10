from image_utils import *
import numpy as np
from random import shuffle

def get_datasets(train_size, val_size, test_size, f_get_dataset):
	train_data = f_get_dataset(train_size)
	val_data = f_get_dataset(val_size)
	test_data = f_get_dataset(test_size)

	return train_data, val_data, test_data

def get_dataset_balanced(size):
	# np.random.seed(seed)

	images = []

	for i in range(size):
		images.append(get_image())

	shuffle(images)

	return images

# Review this
# def get_dataset_unbalanced(size, least_freq_shape=SHAPE_CIRCLE, least_freq_ratio=0.1):
# 	# np.random.seed(seed)

# 	images = []

# 	n_unfreq_shapes = least_freq_ratio * size

# 	for i in range(size):
# 		if i < n_unfreq_shapes:
# 			shape = least_freq_shape
# 		else:
# 			shape = least_freq_shape + 1 if np.random.randint(2) == 0 else least_freq_shape + 2

# 		images.append(get_image(shape))

# 	shuffle(images)

# 	return images

# Only change location
def get_dataset_different_targets(size):
	images = []

	for i in range(size):
		# np.random.seed(seed+i)

		shape = np.random.randint(N_SHAPES)
		color = np.random.randint(N_COLORS)
		size = np.random.randint(N_SIZES)
		img1 = get_image(shape, color, size)

		# Different location
		img2 = get_image(shape, color, size)

		while img1.metadata == img2.metadata:
			img2 = get_image(shape, color, size)

		images.append((img1, img2))


	shuffle(images)

	return images


# Only change location
def get_dataset_dummy_different_targets(size):
	images = []

	for i in range(size):
		# np.random.seed(seed+i)

		shape = np.random.randint(N_SHAPES)
		color = np.random.randint(N_COLORS)
		size = np.random.randint(N_SIZES)
		img1 = get_image(shape, color, size)

		images.append((img1, img1))


	shuffle(images)

	return images


