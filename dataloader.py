import pickle
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from ImageDataset import ImageDataset, ImagesSampler

def load_dictionaries():
	with open("data/mscoco/dict.pckl", "rb") as f:
	    d = pickle.load(f)
	    word_to_idx = d["word_to_idx"] #dictionary w->i
	    idx_to_word = d["idx_to_word"] #list of words
	    bound_idx = word_to_idx["<S>"] # last word in vocab

	return word_to_idx, idx_to_word, bound_idx


def load_data(batch_size, k):
	train_features = np.load('data/mscoco/train_features.npy')
	valid_features = np.load('data/mscoco/valid_features.npy')
	test_features = np.load('data/mscoco/test_features.npy')
	# 2d arrays of 4096 features

	n_image_features = valid_features.shape[1] # 4096

	train_dataset = ImageDataset(train_features)
	valid_dataset = ImageDataset(valid_features, mean=train_dataset.mean, std=train_dataset.std) # All features are normalized with mean and std
	test_dataset = ImageDataset(test_features, mean=train_dataset.mean, std=train_dataset.std)

	train_data = DataLoader(train_dataset, num_workers=8, pin_memory=True, 
		batch_sampler=BatchSampler(ImagesSampler(train_dataset, k, shuffle=True), batch_size=batch_size, drop_last=True))

	valid_data = DataLoader(valid_dataset, num_workers=8, pin_memory=True,
		batch_sampler=BatchSampler(ImagesSampler(valid_dataset, k, shuffle=False), batch_size=batch_size, drop_last=True))

	test_data = DataLoader(test_dataset, num_workers=8, pin_memory=True,
		batch_sampler=BatchSampler(ImagesSampler(test_dataset, k, shuffle=False), batch_size=batch_size, drop_last=True))

	return n_image_features, train_data, valid_data, test_data


def load_shapes_data(batch_size, k):
	folder = 'balanced'

	train_features = np.load('data/shapes/{}/train_features.npy'.format(folder)) #'data/train.large.input.npy'
	valid_features = np.load('data/shapes/{}/valid_features.npy'.format(folder))
	test_features = np.load('data/shapes/{}/test_features.npy'.format(folder))

	train_features = train_features.astype(np.float32)
	valid_features = valid_features.astype(np.float32)
	test_features = test_features.astype(np.float32)

	n_train_examples = len(train_features)
	n_val_examples = len(valid_features)
	n_test_examples = len(test_features)

	train_features = np.reshape(train_features, (n_train_examples, -1))
	valid_features = np.reshape(valid_features, (n_val_examples, -1))
	test_features = np.reshape(test_features, (n_test_examples, -1))

	n_image_features = valid_features.shape[1]

	train_dataset = ImageDataset(train_features)
	valid_dataset = ImageDataset(valid_features, mean=train_dataset.mean, std=train_dataset.std) # All features are normalized with mean and std
	test_dataset = ImageDataset(test_features, mean=train_dataset.mean, std=train_dataset.std)

	train_data = DataLoader(train_dataset, num_workers=8, pin_memory=True, 
		batch_sampler=BatchSampler(ImagesSampler(train_dataset, k, shuffle=True), batch_size=batch_size, drop_last=True))

	valid_data = DataLoader(valid_dataset, num_workers=8, pin_memory=True,
		batch_sampler=BatchSampler(ImagesSampler(valid_dataset, k, shuffle=False), batch_size=batch_size, drop_last=True))

	test_data = DataLoader(test_dataset, num_workers=8, pin_memory=True,
		batch_sampler=BatchSampler(ImagesSampler(test_dataset, k, shuffle=False), batch_size=batch_size, drop_last=True))

	return n_image_features, train_data, valid_data, test_data


