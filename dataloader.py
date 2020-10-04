import pickle
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from ImageDataset import ImageDataset, ImageDatasetSmart, ImageFeaturesDataset, ImagesSampler, ImageFeaturesDatasetZeroShot, ImagesSamplerZeroShot

def load_dictionaries(folder, vocab_size):
	with open("data/{}/dict_{}.pckl".format(folder, vocab_size), "rb") as f:
		d = pickle.load(f)
		word_to_idx = d["word_to_idx"] # dictionary w->i
		idx_to_word = d["idx_to_word"] # list of words
		bound_idx = word_to_idx["<S>"] # last word in vocab

	return word_to_idx, idx_to_word, bound_idx

def load_images(folder, batch_size, k):
	train_filename = '{}/train.large.input.npy'.format(folder)
	valid_filename = '{}/val.input.npy'.format(folder)
	test_filename = '{}/test.input.npy'.format(folder)
	noise_filename = '{}/noise.input.npy'.format(folder)
	train_metadata = pickle.load(open('{}/train.large.metadata.p'.format(folder), 'rb'))
	valid_metadata = pickle.load(open('{}/val.metadata.p'.format(folder), 'rb'))
	test_metadata = pickle.load(open('{}/test.metadata.p'.format(folder), 'rb'))
	noise_metadata = pickle.load(open('{}/noise.metadata.p'.format(folder), 'rb'))

	train_dataset = ImageDataset(train_filename)
	valid_dataset = ImageDataset(valid_filename, mean=train_dataset.mean, std=train_dataset.std) # All features are normalized with mean and std
	test_dataset = ImageDataset(test_filename, mean=train_dataset.mean, std=train_dataset.std)
	noise_dataset = ImageDataset(noise_filename, mean=train_dataset.mean, std=train_dataset.std)

	train_data = DataLoader(train_dataset, num_workers=1, pin_memory=True,
		batch_sampler=BatchSampler(ImagesSampler(train_dataset, train_metadata, k, shuffle=True), batch_size=batch_size, drop_last=False))

	valid_data = DataLoader(valid_dataset, num_workers=1, pin_memory=True,
		batch_sampler=BatchSampler(ImagesSampler(valid_dataset, valid_metadata, k, shuffle=False), batch_size=batch_size, drop_last=False))

	test_data = DataLoader(test_dataset, num_workers=1, pin_memory=True,
		batch_sampler=BatchSampler(ImagesSampler(test_dataset, test_metadata, k, shuffle=False), batch_size=batch_size, drop_last=False))

	noise_data = DataLoader(noise_dataset, num_workers=1, pin_memory=True,
		batch_sampler=BatchSampler(ImagesSampler(noise_dataset, noise_metadata, k, shuffle=False), batch_size=batch_size, drop_last=False))

	return train_data, valid_data, test_data, noise_data

def load_images_smart(folder, batch_size, k):
	train_filename = '{}/train'.format(folder)
	valid_filename = '{}/val'.format(folder)
	test_filename = '{}/test'.format(folder)
	noise_filename = '{}/noise'.format(folder)
	train_dataset = ImageDatasetSmart(train_filename)
	valid_dataset = ImageDatasetSmart(valid_filename) # All features are normalized with mean and std
	test_dataset = ImageDatasetSmart(test_filename)
	noise_dataset = ImageDatasetSmart(noise_filename)

	train_data = DataLoader(train_dataset, num_workers=1, pin_memory=True,
		batch_sampler=BatchSampler(ImagesSampler(train_dataset, k, shuffle=True), batch_size=batch_size, drop_last=False))

	valid_data = DataLoader(valid_dataset, num_workers=1, pin_memory=True,
		batch_sampler=BatchSampler(ImagesSampler(valid_dataset, k, shuffle=False), batch_size=batch_size, drop_last=False))

	test_data = DataLoader(test_dataset, num_workers=1, pin_memory=True,
		batch_sampler=BatchSampler(ImagesSampler(test_dataset, k, shuffle=False), batch_size=batch_size, drop_last=False))

	noise_data = DataLoader(noise_dataset, num_workers=1, pin_memory=True,
		batch_sampler=BatchSampler(ImagesSampler(noise_dataset, k, shuffle=False), batch_size=batch_size, drop_last=False))

	return train_data, valid_data, test_data, noise_data


# This is for loading previously obtained features
def load_pretrained_features(folder,shapes_dataset, batch_size, k, use_symbolic=False):
	if use_symbolic:
		train_features = np.load('{}/train.large.onehot_metadata.p'.format(folder)).astype(np.float32)
		valid_features = np.load('{}/val.onehot_metadata.p'.format(folder)).astype(np.float32)
		test_features = np.load('{}/test.onehot_metadata.p'.format(folder)).astype(np.float32)
		test_features = np.load('{}/noise.onehot_metadata.p'.format(folder)).astype(np.float32)
	else:
		train_features = np.load('{}/train_features.npy'.format(folder))
		valid_features = np.load('{}/valid_features.npy'.format(folder))
		test_features = np.load('{}/test_features.npy'.format(folder))
		noise_features = np.load('{}/noise_features.npy'.format(folder))

	n_image_features = valid_features.shape[-1] # 4096

	train_dataset = ImageFeaturesDataset(train_features)
	valid_dataset = ImageFeaturesDataset(valid_features, mean=train_dataset.mean, std=train_dataset.std) # All features are normalized with mean and std
	test_dataset = ImageFeaturesDataset(test_features, mean=train_dataset.mean, std=train_dataset.std)
	noise_dataset = ImageFeaturesDataset(noise_features, mean=train_dataset.mean, std=train_dataset.std)

	train_metadata = pickle.load(open('shapes/{}/train.large.onehot_metadata.p'.format(shapes_dataset), 'rb'))
	valid_metadata = pickle.load(open('shapes/{}/val.onehot_metadata.p'.format(shapes_dataset), 'rb'))
	test_metadata = pickle.load(open('shapes/{}/test.onehot_metadata.p'.format(shapes_dataset), 'rb'))
	noise_metadata = pickle.load(open('shapes/{}/noise.onehot_metadata.p'.format(shapes_dataset), 'rb'))

	train_data = DataLoader(train_dataset, num_workers=8, pin_memory=True,
		batch_sampler=BatchSampler(ImagesSampler(train_dataset, train_metadata, k, shuffle=True), batch_size=batch_size, drop_last=False))

	valid_data = DataLoader(valid_dataset, num_workers=8, pin_memory=True,
		batch_sampler=BatchSampler(ImagesSampler(valid_dataset, valid_metadata, k, shuffle=False), batch_size=batch_size, drop_last=False))

	test_data = DataLoader(test_dataset, num_workers=8, pin_memory=True,
		batch_sampler=BatchSampler(ImagesSampler(test_dataset, test_metadata, k, shuffle=False), batch_size=batch_size, drop_last=False))

	noise_data = DataLoader(noise_dataset, num_workers=8, pin_memory=True,
		batch_sampler=BatchSampler(ImagesSampler(noise_dataset, noise_metadata, k, shuffle=False), batch_size=batch_size, drop_last=False))

	return n_image_features, train_data, valid_data, test_data, noise_data


# This is for loading previously obtained features
# This needs to grab targets from the unseen dataset and distractors from unseen + seen grabbed uniformly
def load_pretrained_features_zero_shot(target_folder, distractors_folder, shapes_dataset, batch_size, k):
	target_train_features = np.load('{}/train_features.npy'.format(target_folder))
	target_valid_features = np.load('{}/valid_features.npy'.format(target_folder))
	target_test_features = np.load('{}/test_features.npy'.format(target_folder))

	distractors_train_features = np.load('{}/train_features.npy'.format(distractors_folder))
	distractors_valid_features = np.load('{}/valid_features.npy'.format(distractors_folder))
	distractors_test_features = np.load('{}/test_features.npy'.format(distractors_folder))

	n_image_features = target_valid_features.shape[-1]

	assert target_valid_features.shape[-1] == distractors_valid_features.shape[-1]

	train_metadata = pickle.load(open('shapes/{}/train.large.metadata.p'.format(shapes_dataset), 'rb'))
	valid_metadata = pickle.load(open('shapes/{}/val.metadata.p'.format(shapes_dataset), 'rb'))
	test_metadata = pickle.load(open('shapes/{}/test.metadata.p'.format(shapes_dataset), 'rb'))
	noise_metadata = pickle.load(open('shapes/{}/noise.metadata.p'.format(shapes_dataset), 'rb'))

	train_dataset = ImageFeaturesDatasetZeroShot(target_train_features, distractors_train_features)
	valid_dataset = ImageFeaturesDatasetZeroShot(target_valid_features, distractors_valid_features, mean=train_dataset.mean, std=train_dataset.std) # All features are normalized with mean and std
	test_dataset = ImageFeaturesDatasetZeroShot(target_test_features, distractors_test_features, mean=train_dataset.mean, std=train_dataset.std)

	train_data = DataLoader(train_dataset, num_workers=8, pin_memory=True,
		batch_sampler=BatchSampler(ImagesSamplerZeroShot(train_dataset, k, shuffle=True), batch_size=batch_size, drop_last=False))

	valid_data = DataLoader(valid_dataset, num_workers=8, pin_memory=True,
		batch_sampler=BatchSampler(ImagesSamplerZeroShot(valid_dataset, k, shuffle=False), batch_size=batch_size, drop_last=False))

	test_data = DataLoader(test_dataset, num_workers=8, pin_memory=True,
		batch_sampler=BatchSampler(ImagesSamplerZeroShot(test_dataset, k, shuffle=False), batch_size=batch_size, drop_last=False))

	return n_image_features, train_data, valid_data, test_data
