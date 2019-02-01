import pickle
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from ImageDataset import ImageDataset, ImagesSampler

def load_data(batch_size, k):
	with open("data/mscoco/dict.pckl", "rb") as f:
	    d = pickle.load(f)
	    word_to_idx = d["word_to_idx"] #dictionary w->i
	    idx_to_word = d["idx_to_word"] #list of words
	    bound_idx = word_to_idx["<S>"] # last word in vocab

	train_features = np.load('data/mscoco/train_features.npy')
	valid_features = np.load('data/mscoco/valid_features.npy')
	test_features = np.load('data/mscoco/test_features.npy')
	# 2d arrays of 4096 features

	vocab_size = len(word_to_idx) # 10000
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

	return word_to_idx, idx_to_word, bound_idx, vocab_size, n_image_features, train_data, valid_data, test_data


# def load_shapes_data():
	