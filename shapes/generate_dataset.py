from shapes.image_utils import *
import numpy as np
from random import shuffle
from PIL import Image
import os
import pickle
import time
from shapes.bullet_image_utils import get_image as get_image_bullet

def activate_bullet():
	global get_image
	get_image = get_image_bullet

def get_shape_probs():
	assert N_SHAPES == 3

	probs = np.random.dirichlet(np.ones(3)*10)
	diff1 = abs(probs[0] - probs[1])
	diff2 = abs(probs[1] - probs[2])
	diff3 = abs(probs[2] - probs[0])

	while (diff1 > 0.2 or diff2 > 0.2 or diff3 > 0.2
		or (diff1 < 0.1 and diff2 < 0.1 and diff3 < 0.1)):
		probs = np.random.dirichlet(np.ones(3))
		diff1 = abs(probs[0] - probs[1])
		diff2 = abs(probs[1] - probs[2])
		diff3 = abs(probs[2] - probs[0])

	return probs

# P(red|circle)
# P(green|circle)
# P(blue|circle)
# P(red|square)
# P(green|square)
# P(blue|square)
# P(red|triangle)
# P(green|triangle)
# P(blue|triangle)
def get_color_given_shape_probs(n_colors=3):
	probs = []
	for s in range(N_SHAPES):
		shape_probs = np.random.dirichlet(np.ones(n_colors)/10)
		if n_colors < 3: # Exclude red triangle, blue square, green circle
			if s == SHAPE_TRIANGLE:
				shape_probs = [0.0, shape_probs[0], shape_probs[1]]
			elif s == SHAPE_SQUARE:
				shape_probs = [shape_probs[0], shape_probs[1], 0.0]
			else: # SHAPE_CIRCLE
				shape_probs = [shape_probs[0], 0.0, shape_probs[1]]

		probs.extend(shape_probs)

	return probs

def get_datasets_smart(train_size, val_size, test_size, f_get_dataset, is_uneven, folder_name):
	real_sizes = [train_size, val_size, test_size, val_size]
	paths = ['train','val', 'test', 'noise']
	if is_uneven:
		shapes_probs = get_shape_probs()
		if f_get_dataset is get_dataset_uneven_incomplete or f_get_dataset is get_dataset_uneven_different_targets_row_incomplete:
			color_given_shape_probs = get_color_given_shape_probs(n_colors=2)
		else:
			color_given_shape_probs = get_color_given_shape_probs()
		for typ, real_size in enumerate(real_sizes):
			for i in range(real_size):
				data = f_get_dataset(1, shapes_probs, color_given_shape_probs)[0]
				has_tuples = type(data) is tuple
				if not has_tuples:
					image_data = np.asarray(data.data[:,:,0:3])
				else:
					image_data = np.array([data[0].data[:,:,0:3], data[1].data[:,:,0:3]])
				np.save("{}/{}_{}.input".format(folder_name, paths[typ], i), image_data)
				pickle.dump(data.metadata, open('{}/{}_{}.metadata.p'.format(folder_name, paths[typ], i), 'wb'))

		return None, None, None, shapes_probs, color_given_shape_probs
	else:
		for typ, real_size in enumerate(real_sizes):
			for i in range(real_size):
				data = f_get_dataset(1)[0]
				has_tuples = type(data) is tuple
				if not has_tuples:
					image_data = np.asarray(data.data[:,:,0:3])
				else:
					image_data = np.array([data[0].data[:,:,0:3], data[1].data[:,:,0:3]])
				np.save("{}/{}_{}.input".format(folder_name, paths[typ], i), image_data)
				pickle.dump(data.metadata, open('{}/{}_{}.metadata.p'.format(folder_name, paths[typ], i), 'wb'))

		return None, None, None

def get_datasets(train_size, val_size, test_size, f_get_dataset, is_uneven):
	if is_uneven:
		shapes_probs = get_shape_probs()
		if f_get_dataset is get_dataset_uneven_incomplete or f_get_dataset is get_dataset_uneven_different_targets_row_incomplete:
			color_given_shape_probs = get_color_given_shape_probs(n_colors=2)
		else:
			color_given_shape_probs = get_color_given_shape_probs()

		train_data = f_get_dataset(train_size, shapes_probs, color_given_shape_probs)
		val_data = f_get_dataset(val_size, shapes_probs, color_given_shape_probs)
		test_data = f_get_dataset(test_size, shapes_probs, color_given_shape_probs)

		return train_data, val_data, test_data, shapes_probs, color_given_shape_probs
	else:
		train_data = f_get_dataset(train_size)
		val_data = f_get_dataset(val_size)
		test_data = f_get_dataset(test_size)

		return train_data, val_data, test_data

def get_dataset_balanced(dataset_size):
	images = []

	for i in range(dataset_size):
		images.append(get_image())

	shuffle(images)

	return images

# Condition color given shape
def get_dataset_uneven(dataset_size, shapes_probs, color_given_shape_probs):
	images = []

	for i in range(dataset_size):
		shape = np.random.choice(range(N_SHAPES), p=shapes_probs)
		color_probs = color_given_shape_probs[shape*N_SHAPES:shape*N_SHAPES+N_COLORS]
		color = np.random.choice(range(N_COLORS), p=color_probs)

		images.append(get_image([Figure(shape, color, size=-1, r=-1, c=-1)]))

	shuffle(images)

	return images

# Only change location
def get_dataset_different_targets(dataset_size):
	images = []

	for i in range(dataset_size):
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


def get_dataset_different_targets_two_figures(dataset_size):
	images = []
	n_figures = 2

	for i in range(dataset_size):

		figures_orig = []
		past_locations = [] # tuples (r,c)
		for i in range(n_figures):
			shape = np.random.randint(N_SHAPES)
			color = np.random.randint(N_COLORS)
			size = np.random.randint(N_SIZES)
			r = np.random.randint(N_CELLS)
			c = np.random.randint(N_CELLS)

			if i > 0: # Check for collisions
				while (r,c) in past_locations:
					r = np.random.randint(N_CELLS)
					c = np.random.randint(N_CELLS)

			past_locations.append((r,c))

			figures_orig.append(Figure(shape, color, size, r, c))


		figures_new = []

		# Sort by location
		figures_orig.sort(key=lambda x: x.r)

		# for i, fig in enumerate(figures_orig):
		fig = figures_orig[0]

		shape = fig.shape
		color = fig.color
		size = fig.size

		is_same_row = fig.r == figures_orig[-1].r

		if is_same_row:
			r = fig.r
		else:
			r = np.random.randint(0, N_CELLS - 1) # leave at least the last row free

		is_same_col = fig.c == figures_orig[-1].c

		if is_same_col:
			c = fig.c
		else:
			is_right = fig.c - figures_orig[-1].c > 0
			if is_right:
				c = np.random.randint(1, N_CELLS) # leave at least the first column free
			else:
				c = np.random.randint(0, N_CELLS -1) # leave at least the last column free

		figures_new.append(Figure(shape, color, size, r, c))

		shape2 = figures_orig[-1].shape
		color2 = figures_orig[-1].color
		size2 = figures_orig[-1].size

		if is_same_row:
			r2 = r
		else:
			r2 = np.random.randint(r + 1, N_CELLS)

		if is_same_col:
			c2 = c
		else:
			if is_right:
				if c == 1:
					c2 = 0
				else:
					c2 = np.random.randint(0, c - 1)
			else:
				if c + 1 == N_CELLS:
					c2 = N_CELLS - 1
				else:
					c2 = np.random.randint(c + 1, N_CELLS)

		figures_new.append(Figure(shape2, color2, size2, r2, c2))

		# Actually generate the images
		img1 = get_image(figures_orig)
		img2 = get_image(figures_new)

		images.append((img1, img2))

	shuffle(images)

	return images


def calculate_col(fig0, fig1, fig2):
	is_left_most = fig0.c <= fig1.c and fig0.c <= fig2.c

	if is_left_most:
		# print("left most")
		if fig1.c <= fig2.c:
			c0 = np.random.randint(0, fig1.c) if fig1.c > 0 else 0
		else:
			c0 = np.random.randint(0, fig2.c) if fig2.c > 0 else 0
	else:
		is_right_most = fig0.c >= fig1.c and fig0.c >= fig2.c

		if is_right_most:
			# print("right most")
			if fig1.c >= fig2.c:
				c0 = np.random.randint(fig1.c + 1, N_CELLS) if fig1.c+1 < N_CELLS else N_CELLS -1
			else:
				c0 = np.random.randint(fig2.c + 1, N_CELLS) if fig2.c+1 < N_CELLS else N_CELLS -1

		else: #I'm middle
			# print("Middle")
			if fig1.c >= fig2.c:
				c0 = np.random.randint(fig2.c + 1, fig1.c) if fig2.c+1 < fig1.c else fig2.c +1
			else:
				c0 = np.random.randint(fig1.c + 1, fig2.c) if fig1.c+1 < fig2.c else fig1.c +1

	return c0


def get_dataset_different_targets_three_figures(dataset_size):
	images = []
	n_figures = 3

	for i in range(dataset_size):
		figures_orig = []
		past_locations = [] # tuples (r,c)
		for i in range(n_figures):
			shape = np.random.randint(N_SHAPES)
			color = np.random.randint(N_COLORS)
			size = np.random.randint(N_SIZES)
			r = np.random.randint(N_CELLS)
			c = np.random.randint(N_CELLS)

			if i > 0: # Check for collisions
				while (r,c) in past_locations:
					r = np.random.randint(N_CELLS)
					c = np.random.randint(N_CELLS)

			past_locations.append((r,c))

			figures_orig.append(Figure(shape, color, size, r, c))

		# figures_orig= [Figure(0,0,0,0,0), Figure(0,0,0,5,2), Figure(0,0,0,8,1)]

		figures_new = []

		# Sort by location
		figures_orig.sort(key=lambda x: x.r * N_CELLS + x.c)

		fig0 = figures_orig[0]
		fig1 = figures_orig[1]
		fig2 = figures_orig[2]

		# print(figures_orig)

		is_same_row = fig0.r == fig1.r

		if is_same_row:
			r0 = fig0.r
		else:
			r0 = np.random.randint(0, fig1.r) # between my row and the next's

		is_same_col = fig0.c == fig1.c

		if is_same_col:
			c0 = fig0.c
		else:
			# print("c0")
			c0 = calculate_col(fig0, fig1, fig2)

		if is_same_row:
			r1 = r0
		else:
			r1 = np.random.randint(r0 + 1, fig2.r) if r0+1 < fig2.r else r0+1

		if is_same_col:
			c1 = c0
		else:
			# print("c1")
			c1 = calculate_col(fig1, fig0, fig2)


		if c1 == c0 and is_same_row:
			if fig0.c > fig1.c:
				if c1 >= 1:
					c1 -= 1
				else:
					c0 += 1
			else:
				if c0 >= 1:
					c0 -=1
				else:
					c1 += 1

		is_same_row = fig1.r == fig2.r

		if is_same_row:
			r2 = r1
		else:
			r2 = np.random.randint(r1+1, N_CELLS) if r1+1 < N_CELLS else N_CELLS-1

		is_same_col = fig1.c == fig2.c

		if is_same_col:
			c2 = c1
		else:
			# print("c2")
			c2 = calculate_col(fig2, fig0, fig1)


		if c1 == c2 and is_same_row:
			if fig2.c > fig1.c:
				if c1 >= 1:
					c1 -= 1
				else:
					c2 += 1
			else:
				if c2 >= 1:
					c2 -=1
				else:
					c1 += 1


		figures_new.append(Figure(fig0.shape, fig0.color, fig0.size, r0, c0))
		figures_new.append(Figure(fig1.shape, fig1.color, fig1.size, r1, c1))
		figures_new.append(Figure(fig2.shape, fig2.color, fig2.size, r2, c2))

		# print("fig0", r0, c0)
		# print("fig1", r1, c1)
		# print("fig2", r2, c2)

		# Actually generate the images
		img1 = get_image(figures_orig)
		img2 = get_image(figures_new)

		images.append((img1, img2))

	shuffle(images)

	return images

# Exclude red triangle, blue square, green circle
def get_dataset_balanced_incomplete(dataset_size):
	images = []

	for i in range(dataset_size):
		shape = np.random.randint(N_SHAPES)
		if shape == SHAPE_TRIANGLE:
			color = COLOR_GREEN if np.random.randint(2) == 0 else COLOR_BLUE
		elif shape == SHAPE_SQUARE:
			color = COLOR_RED if np.random.randint(2) == 0 else COLOR_GREEN
		else: #SHAPE_CIRCLE
			color = COLOR_BLUE if np.random.randint(2) == 0 else COLOR_RED

		images.append(get_image([Figure(shape, color, size=-1, r=-1, c=-1)]))

	shuffle(images)

	return images

# Exclude red triangle, blue square, green circle
# Only change location
def get_dataset_different_targets_incomplete(dataset_size):
	images = []

	for i in range(dataset_size):
		shape = np.random.randint(N_SHAPES)
		if shape == SHAPE_TRIANGLE:
			color = COLOR_GREEN if np.random.randint(2) == 0 else COLOR_BLUE
		elif shape == SHAPE_SQUARE:
			color = COLOR_RED if np.random.randint(2) == 0 else COLOR_GREEN
		else: #SHAPE_CIRCLE
			color = COLOR_BLUE if np.random.randint(2) == 0 else COLOR_RED

		size = np.random.randint(N_SIZES)

		img1 = get_image([Figure(shape, color, size, r=-1, c=-1)])

		# Different location
		img2 = get_image([Figure(shape, color, size, r=-1, c=-1)])

		while img1.metadata == img2.metadata:
			img2 = get_image([Figure(shape, color, size, r=-1, c=-1)])

		images.append((img1, img2))


	shuffle(images)

	return images

# Exclude red triangle, blue square, green circle
# Condition color given shape
def get_dataset_uneven_incomplete(dataset_size, shapes_probs, color_given_shape_probs):
	images = []

	for i in range(dataset_size):
		shape = np.random.choice(range(N_SHAPES), p=shapes_probs)
		color_probs = color_given_shape_probs[shape*N_SHAPES:shape*N_SHAPES+N_COLORS]
		color = np.random.choice(range(N_COLORS), p=color_probs)

		images.append(get_image([Figure(shape, color, size=-1, r=-1, c=-1)]))

	shuffle(images)

	return images

# Exclude red triangle, blue square, green circle
def get_dataset_uneven_different_targets_row_incomplete(dataset_size, shapes_probs, color_given_shape_probs):
	images = []

	for i in range(dataset_size):
		shape = np.random.choice(range(N_SHAPES), p=shapes_probs)
		color_probs = color_given_shape_probs[shape*N_SHAPES:shape*N_SHAPES+N_COLORS]
		color = np.random.choice(range(N_COLORS), p=color_probs)
		size = np.random.randint(N_SIZES)
		column = np.random.randint(N_CELLS)

		img1 = get_image([Figure(shape, color, size, r=-1, c=column)])

		# Different row
		img2 = get_image([Figure(shape, color, size, r=-1, c=column)])


		while img1.metadata == img2.metadata:
			img2 = get_image([Figure(shape, color, size, r=-1, c=column)])
		images.append((img1, img2))


	shuffle(images)

	return images


# Only change location
def get_dataset_uneven_different_targets(dataset_size, shapes_probs, color_given_shape_probs):
	images = []

	for i in range(dataset_size):
		shape = np.random.choice(range(N_SHAPES), p=shapes_probs)
		color_probs = color_given_shape_probs[shape*N_SHAPES:shape*N_SHAPES+N_COLORS]
		color = np.random.choice(range(N_COLORS), p=color_probs)
		size = np.random.randint(N_SIZES)

		img1 = get_image([Figure(shape, color, size, r=-1, c=-1)])

		# Different location
		img2 = get_image([Figure(shape, color, size, r=-1, c=-1)])

		while img1.metadata == img2.metadata:
			img2 = get_image([Figure(shape, color, size, r=-1, c=-1)])

		images.append((img1, img2))


	shuffle(images)

	return images

# Only change row
def get_dataset_uneven_different_targets_row(dataset_size, shapes_probs, color_given_shape_probs):
	images = []

	for i in range(dataset_size):
		shape = np.random.choice(range(N_SHAPES), p=shapes_probs)
		color_probs = color_given_shape_probs[shape*N_SHAPES:shape*N_SHAPES+N_COLORS]
		color = np.random.choice(range(N_COLORS), p=color_probs)
		size = np.random.randint(N_SIZES)
		column = np.random.randint(N_CELLS)

		img1 = get_image([Figure(shape, color, size, r=-1, c=column)])

		# Different row
		img2 = get_image([Figure(shape, color, size, r=-1, c=column)])

		while img1.metadata == img2.metadata:
			img2 = get_image([Figure(shape, color, size, r=-1, c=column)])

		images.append((img1, img2))


	shuffle(images)

	return images

# Only change size
def get_dataset_uneven_different_targets_size(dataset_size, shapes_probs, color_given_shape_probs):
	images = []

	for i in range(dataset_size):
		shape = np.random.choice(range(N_SHAPES), p=shapes_probs)
		color_probs = color_given_shape_probs[shape*N_SHAPES:shape*N_SHAPES+N_COLORS]
		color = np.random.choice(range(N_COLORS), p=color_probs)
		row = np.random.randint(N_CELLS)
		column = np.random.randint(N_CELLS)

		img1 = get_image([Figure(shape, color, size=-1, r=row, c=column)])

		# Different location
		img2 = get_image([Figure(shape, color, size=-1, r=row, c=column)])

		while img1.metadata == img2.metadata:
			img2 = get_image([Figure(shape, color, size=-1, r=row, c=column)])

		images.append((img1, img2))


	shuffle(images)

	return images


# Only have red triangle, blue square, green circle
def get_dataset_balanced_zero_shot(dataset_size):
	images = []

	for i in range(dataset_size):
		shape = np.random.randint(N_SHAPES)
		if shape == SHAPE_TRIANGLE:
			color = COLOR_RED
		elif shape == SHAPE_SQUARE:
			color = COLOR_BLUE
		else: #SHAPE_CIRCLE
			color = COLOR_GREEN

		images.append(get_image([Figure(shape, color, size=-1, r=-1, c=-1)]))

	shuffle(images)

	return images


# Only have red triangle, blue square, green circle
# Only change location
def get_dataset_different_targets_zero_shot(dataset_size):
	images = []

	for i in range(dataset_size):
		shape = np.random.randint(N_SHAPES)
		if shape == SHAPE_TRIANGLE:
			color = COLOR_RED
		elif shape == SHAPE_SQUARE:
			color = COLOR_BLUE
		else: #SHAPE_CIRCLE
			color = COLOR_GREEN

		size = np.random.randint(N_SIZES)

		img1 = get_image([Figure(shape, color, size, r=-1, c=-1)])

		# Different location
		img2 = get_image([Figure(shape, color, size, r=-1, c=-1)])

		while img1.metadata == img2.metadata:
			img2 = get_image([Figure(shape, color, size, r=-1, c=-1)])

		images.append((img1, img2))


	shuffle(images)

	return images

# Only have red triangle, blue square, green circle
# Only change row



# def get_dataset_dummy_different_targets(size):
# 	images = []

# 	for i in range(size):
# 		# np.random.seed(seed+i)

# 		shape = np.random.randint(N_SHAPES)
# 		color = np.random.randint(N_COLORS)
# 		size = np.random.randint(N_SIZES)
# 		img1 = get_image(shape, color, size)

# 		images.append((img1, img1))


# 	shuffle(images)

# 	return images
