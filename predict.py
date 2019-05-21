import pickle
import numpy as np
import random
from datetime import datetime
import os
import sys
import time
import math

import torch
from model import Model
from run import evaluate
from dataloader import load_dictionaries, load_images, load_pretrained_features
from build_shapes_dictionaries import *
from metadata import does_shapes_onehot_metadata_exist, create_shapes_onehot_metadata, load_shapes_onehot_metadata
from decode import dump_words
from visual_module import CNN
from dump_cnn_features import save_features
# from rsa import representation_similarity_analysis_freq


use_gpu = torch.cuda.is_available()
debugging = not use_gpu
should_dump = True#not debugging
should_covert_to_words = not debugging
should_dump_indices = not debugging


EPOCHS = 60 if not debugging else 2
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
BATCH_SIZE = 128 if not debugging else 8
K = 3  # number of distractors
n_image_features = 2048#4096

# Default settings
vocab_size = 10
max_sentence_length = 5
shapes_dataset = 'balanced'
vl_loss_weight = 0.0
bound_weight = 1.0
should_train_visual = False
model_file_name = None
cnn_model_file_name = None
rsa_sampling = -1
seed = 42

# Overwrite default settings if given in command line
if len(sys.argv) > 1:
	seed = int(sys.argv[1])
	vocab_size = int(sys.argv[2])
	max_sentence_length = int(sys.argv[3])
	shapes_dataset = sys.argv[4]
	vl_loss_weight = float(sys.argv[5])
	bound_weight = float(sys.argv[6])
	model_file_name = sys.argv[7]
	cnn_model_file_name = sys.argv[8] # dumps/0408134521185696/0408134521185696_41_model
	rsa_sampling = int(sys.argv[9])

#assert should_train_visual or cnn_model_file_name is not None, 'Need stored CNN weights if not training visual features'

# Get model id using a timestamp
dump_id = '{:%m%d%H%M%S%f}'.format(datetime.now())


################# Print info ####################
print('========================================')
print('Dump id: {}'.format(dump_id))
print('Seed: {}'.format(seed))
print('Loading pretrained model from: {}'.format(model_file_name))
print('Loading pretrained CNN from: {}'.format(cnn_model_file_name))
print('|V|: {}'.format(vocab_size))
print('L: {}'.format(max_sentence_length))
print('Using gpu: {}'.format(use_gpu))
print('Dataset: {}'.format(shapes_dataset))
print('Lambda: {}'.format(vl_loss_weight))
print('Alpha: {}'.format(bound_weight))
print('N image features: {}'.format(n_image_features))
if rsa_sampling >= 0:
	print('N samples for RSA: {}'.format(rsa_sampling))
print()
#################################################


# Create vocab if there is not one for the desired size already
if not does_vocab_exist(vocab_size):
	build_vocab(vocab_size)

# Load vocab
word_to_idx, idx_to_word, bound_idx = load_dictionaries('shapes', vocab_size)

# Load/generate features
cnn_dump_id = cnn_model_file_name.split('/')[-1]

features_folder_name = 'data/shapes/{}_{}'.format(shapes_dataset, cnn_dump_id)

# Check if the features were already extracted with this CNN
if not os.path.exists(features_folder_name):
	# Load CNN from dumped model
	state = torch.load(cnn_model_file_name, map_location= lambda storage, location: storage)
	cnn_state = {k[4:]:v for k,v in state.items() if 'cnn' in k}
	trained_cnn = CNN(n_image_features)
	trained_cnn.load_state_dict(cnn_state)

	if use_gpu:
		trained_cnn = trained_cnn.cuda()

	print("=CNN state loaded=")
	print("Extracting features...")

	# Dump the features to then load them
	features_folder_name = save_features(trained_cnn, shapes_dataset, cnn_dump_id)


# Load data
if should_train_visual:
	_train_data, _valid_data, _test_data = load_images('shapes/{}'.format(shapes_dataset), BATCH_SIZE, K)
else:
	n_pretrained_image_features, _train_data, _valid_data, test_data = load_pretrained_features(features_folder_name, BATCH_SIZE, K)
	assert n_pretrained_image_features == n_image_features

# Create onehot metadata if not created yet
if not does_shapes_onehot_metadata_exist(shapes_dataset):
	create_shapes_onehot_metadata(shapes_dataset)

# Load metadata
_train_metadata, _valid_metadata, test_metadata = load_shapes_onehot_metadata(shapes_dataset)


# Settings
dumps_dir = './dumps'
if should_dump and not os.path.exists(dumps_dir):
	os.mkdir(dumps_dir)


current_model_dir = '{}/{}_{}_{}'.format(dumps_dir, dump_id, vocab_size, max_sentence_length)

if should_dump and not os.path.exists(current_model_dir):
	os.mkdir(current_model_dir)


model = Model(n_image_features, vocab_size,
	EMBEDDING_DIM, HIDDEN_SIZE, 
	bound_idx, max_sentence_length, 
	vl_loss_weight, bound_weight, 
	should_train_visual, rsa_sampling,
	use_gpu)

# Load model to evaluate
state = torch.load(model_file_name, map_location= lambda storage, location: storage)
without_cnn_state = {k:v for k,v in state.items() if not 'cnn' in k}
model.load_state_dict(without_cnn_state)

if use_gpu:
	model = model.cuda()

# Evaluate model on test data
test_word_counts = torch.zeros([vocab_size])
if use_gpu:
	test_word_counts = test_word_counts.cuda()

(test_loss_meter, 
test_acc_meter, 
test_messages, 
test_indices,
_w_counts, 
test_entropy_meter,
test_distinctness_meter,
test_rsa_sr_meter,
test_rsa_si_meter,
test_rsa_ri_meter,
test_topological_sim_meter,
test_input_embed_send,
test_input_embed_rec) = evaluate(model, test_data, test_word_counts, test_metadata, debugging)

print()
print('Test accuracy: {}'.format(test_acc_meter.avg))

if should_dump:
	best_epoch = model_file_name.split('_')[-2]

	pickle.dump(test_loss_meter, open('{}/{}_{}_test_losses_meter.p'.format(current_model_dir, dump_id, best_epoch), 'wb'))
	pickle.dump(test_acc_meter, open('{}/{}_{}_test_accuracy_meter.p'.format(current_model_dir, dump_id, best_epoch), 'wb'))
	pickle.dump(test_entropy_meter, open('{}/{}_{}_test_entropy_meter.p'.format(current_model_dir, dump_id, best_epoch), 'wb'))
	pickle.dump(test_distinctness_meter, open('{}/{}_{}_test_distinctness_meter.p'.format(current_model_dir, dump_id, best_epoch), 'wb'))
	pickle.dump(test_rsa_sr_meter, open('{}/{}_{}_test_rsa_sr_meter.p'.format(current_model_dir, dump_id, best_epoch), 'wb'))
	pickle.dump(test_rsa_si_meter, open('{}/{}_{}_test_rsa_si_meter.p'.format(current_model_dir, dump_id, best_epoch), 'wb'))
	pickle.dump(test_rsa_ri_meter, open('{}/{}_{}_test_rsa_ri_meter.p'.format(current_model_dir, dump_id, best_epoch), 'wb'))
	pickle.dump(test_topological_sim_meter, open('{}/{}_{}_test_topological_sim_meter.p'.format(current_model_dir, dump_id, best_epoch), 'wb'))
	pickle.dump(test_messages, open('{}/{}_{}_test_messages.p'.format(current_model_dir, dump_id, best_epoch), 'wb'))

	if should_dump_indices:
		pickle.dump(test_indices, open('{}/{}_{}_test_imageIndices.p'.format(current_model_dir, dump_id, best_epoch), 'wb'))			

	if should_covert_to_words:
		dump_words(current_model_dir, test_messages, idx_to_word, '{}_{}_test_messages'.format(dump_id, best_epoch))

	# if 'uneven' in shapes_dataset:
	# 	# Calculate RSA and topological wrt shape-color frequency
	# 	shape_color_freq, freq_rsa_sr, freq_rsa_si, freq_rsa_ri, freq_topological_similarity = representation_similarity_analysis_freq(
	# 			np.load('{}/test_features.npy'.format(features_folder_name)),
	# 			test_metadata,
	# 			test_messages.cpu().numpy(),
	# 			test_input_embed_send.detach().cpu(),
	# 			test_input_embed_rec.detach().cpu(),
	# 			samples=500
	# 		)

	# 	freq_dict = {
	# 		'shape_color_freq' : shape_color_freq,
	# 		'rsa_sr': freq_rsa_sr,
	# 		'rsa_si': freq_rsa_si,
	# 		'rsa_ri': freq_rsa_ri,
	# 		'topo': freq_topological_similarity,
	# 	}

	# 	pickle.dump(freq_dict, open('{}/{}_{}_test_frequency_based_rsa_topo.p'.format(current_model_dir, dump_id, best_epoch), 'wb'))
			