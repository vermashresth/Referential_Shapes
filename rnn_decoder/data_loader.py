import pickle
import numpy as np
import os
from torch.utils.data import DataLoader


class MessageImageDataset():
	def __init__(self, messages_file_name, images_file_name, image_indices_file_name):
		self.messages = pickle.load(open(messages_file_name, 'rb'))
		self.images = np.load(images_file_name).astype(np.float32)
		self.indices = pickle.load(open(image_indices_file_name, 'rb'))[:,0] # Only grab target!

		if 'cuda' in str(self.messages.device):
			self.messages = self.messages.cpu()
		if 'cuda' in str(self.indices.device):
			self.indices = self.indices.cpu()

	def __getitem__(self, index):
		message = self.messages[index]
		image_idx = self.indices[index]
		image = self.images[image_idx]

		if image.ndim == 4: #diff targets
			image = image[0,:,:,:] # Only grab the target used for generating the message

		return (message, image)

	def __len__(self):
		return len(self.messages)

def get_file_names(model_dir, file_name_id):
	file_names = ['{}/{}'.format(model_dir, f) for f in os.listdir(model_dir) if file_name_id in f]
	
	if len(file_names) > 1:
		# Make sure we want training dumps
		assert '_test_' not in file_name_id and '_eval_' not in file_name_id
		file_names = [f for f in file_names if '_test_' not in f and '_eval_' not in f]
		if 'entropy' in file_name_id:
			if len(file_names) == 2: # language and non language entropies
				if len(file_names[0]) < len(file_names[1]):
					file_names = [file_names[0]]
				else:
					file_names = [file_names[1]]

	assert len(file_names) == 1

	file_name = file_names[0]

	return file_name

def get_model_dir(model_id):
	dumps_dir = '../dumps'
	return '{}/{}'.format(dumps_dir, model_id)

def get_best_epoch(model_dir):
	file_name = get_file_names(model_dir, 'test_losses_meter.p')
	return int(file_name.split('_')[-4])

def get_full_file_names(model_id, file_id, shapes_dataset):
	if file_id == 'train':
		to_append = ''
	else:
		to_append = '{}_'.format(file_id)

	model_dir = get_model_dir(model_id)
	best_epoch = get_best_epoch(model_dir)

	messages_file_name = get_file_names(model_dir, '_{}_{}{}'.format(best_epoch, to_append, 'messages.p'))
	indices_file_name = get_file_names(model_dir, '_{}_{}{}'.format(best_epoch, to_append, 'imageIndices.p'))

	if file_id == 'train':
		to_append = 'train.large'
	elif file_id == 'eval':
		to_append = 'val'
	else:
		to_append = 'test'

	shapes_dir = '../shapes/{}'.format(shapes_dataset)
	images_file_name = get_file_names(shapes_dir, '{}.input.npy'.format(to_append))

	return (messages_file_name, images_file_name, indices_file_name)

def load_message_image_data(model_id, shapes_dataset, batch_size):
	train_messages_file_name, train_images_file_name, train_image_indices_file_name = get_full_file_names(model_id, 'train', shapes_dataset)
	valid_messages_file_name, valid_images_file_name, valid_image_indices_file_name = get_full_file_names(model_id, 'eval', shapes_dataset)
	test_messages_file_name, test_images_file_name, test_image_indices_file_name = get_full_file_names(model_id, 'test', shapes_dataset)

	train_dataset = MessageImageDataset(train_messages_file_name, train_images_file_name, train_image_indices_file_name)
	valid_dataset = MessageImageDataset(valid_messages_file_name, valid_images_file_name, valid_image_indices_file_name)
	test_dataset = MessageImageDataset(test_messages_file_name, test_images_file_name, test_image_indices_file_name)

	train_data = DataLoader(train_dataset, num_workers=1, pin_memory=True, batch_size=batch_size, drop_last=False, shuffle=True)
	valid_data = DataLoader(valid_dataset, num_workers=1, pin_memory=True, batch_size=batch_size, drop_last=False, shuffle=False)
	test_data = DataLoader(test_dataset, num_workers=1, pin_memory=True, batch_size=batch_size, drop_last=False, shuffle=False)

	return train_data, valid_data, test_data