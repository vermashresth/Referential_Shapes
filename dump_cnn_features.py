import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms
from PIL import Image
import torch.utils.data as data
from torch.utils.data import DataLoader
import os


use_gpu = torch.cuda.is_available()


class ShapesDataset(data.Dataset):
	def __init__(self, images_filename, mean=None, std=None):
		super().__init__()

		self.data = np.load(images_filename)

		self.n_tuples = 0 if len(self.data.shape) < 5 else self.data.shape[1]

		if mean is None:
			mean = np.mean(self.data, axis=tuple(range(self.data.ndim-1)))#np.mean(features, axis=0)
			std = np.std(self.data, axis=tuple(range(self.data.ndim-1)))#np.std(features, axis=0)
			std[np.nonzero(std == 0.0)] = 1.0  # nan is because of dividing by zero
		self.mean = mean
		self.std = std

		self.transforms = torchvision.transforms.Compose([
			torchvision.transforms.ToPILImage(),
			torchvision.transforms.Resize((128, 128), Image.LINEAR),
			torchvision.transforms.ToTensor(),
			torchvision.transforms.Normalize(self.mean, self.std)
			])

	def __getitem__(self, index):
		if self.n_tuples == 0:
			image = self.data[index, :, :, :]
			image = self.transforms(image)
		else:
			imgs = []
			for i in range(self.n_tuples):
				img = self.data[index, i, :, :, :]
				img = self.transforms(img)

				imgs.append(img)

			image = torch.stack(imgs) # 2 x 3 x 128 x 128

		return image

	def __len__(self):
		return self.data.shape[0]

def cnn_fwd(model, x):	
	y = model(x)
	y = y.detach()

	if use_gpu:
		y = y.cpu()

	return y.numpy()

def get_features(model, dataloader, output_data_folder, file_id):
	for i, x in enumerate(dataloader):
		if use_gpu:
			x = x.cuda()

		if i == 0:
			if len(x.shape) == 5:
				n_tuples = x.shape[1]
			else:
				n_tuples = 0
		

		if n_tuples == 0:
			y = cnn_fwd(model, x)

		else:
			ys = []
			for j in range(n_tuples):
				x_t = x[:,j,:,:,:]
				y_t = cnn_fwd(model, x_t)
				ys.append(y_t)
			
			# Here we need to combine 1st elem with 1st elem, etc in the batch

			y = np.asarray(ys)
			y = np.moveaxis(y, 0, 1)


		np.save('{}/{}_{}_features.npy'.format(output_data_folder, file_id, i), y)

		if not use_gpu and i == 5:
			break


def stitch_files(temp_folder, output_data_folder, file_id):
	file_names = ['{}/{}'.format(temp_folder, f) for f in os.listdir(temp_folder) if file_id in f]
	file_names.sort(key=os.path.getctime)

	for i, f in enumerate(file_names):
		arr = np.load(f)
		if i == 0:
			features = arr
		else:
			features = np.concatenate((features, arr))

	np.save('{}/{}_features.npy'.format(output_data_folder, file_id), features)


def save_features(cnn, folder, folder_id):
	batch_size = 128 if use_gpu else 4

	train_dataset = ShapesDataset('shapes/{}/train.large.input.npy'.format(folder))
	val_dataset = ShapesDataset('shapes/{}/val.input.npy'.format(folder), mean=train_dataset.mean, std=train_dataset.std)
	test_dataset = ShapesDataset('shapes/{}/test.input.npy'.format(folder), mean=train_dataset.mean, std=train_dataset.std)

	train_dataloader = DataLoader(train_dataset, num_workers=8, batch_size=batch_size)
	val_dataloader = DataLoader(val_dataset, num_workers=8, batch_size=batch_size)
	test_dataloader = DataLoader(test_dataset, num_workers=8, batch_size=batch_size)

	output_features_folder = 'data/shapes/{}_{}'.format(folder, folder_id)
	temp_features_folder = 'data/temp'

	if not os.path.exists(output_features_folder):
		os.mkdir(output_features_folder)

	if not os.path.exists(temp_features_folder):
		os.mkdir(temp_features_folder)

	# Make a pass of the datasets through the trained CNN layers
	cnn.eval()

	get_features(cnn, train_dataloader, temp_features_folder, 'train')
	get_features(cnn, val_dataloader, temp_features_folder, 'valid')
	get_features(cnn, test_dataloader, temp_features_folder, 'test')

	# Stitch into one file
	stitch_files(temp_features_folder, output_features_folder, 'train')
	stitch_files(temp_features_folder, output_features_folder, 'valid')
	stitch_files(temp_features_folder, output_features_folder, 'test')

	# Remove temp folder
	for f in os.listdir(temp_features_folder):
		os.remove('{}/{}'.format(temp_features_folder, f))
	os.rmdir(temp_features_folder)

	print('Visual features saved in folder {}'.format(output_features_folder))
	print()

	return output_features_folder