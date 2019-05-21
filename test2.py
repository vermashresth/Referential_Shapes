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
from run import train_one_epoch, evaluate
from utils import EarlyStopping
from dataloader import load_dictionaries, load_images, load_pretrained_features
from build_shapes_dictionaries import *
from metadata import does_shapes_onehot_metadata_exist, create_shapes_onehot_metadata, load_shapes_onehot_metadata
from decode import dump_words
from visual_module import CNN
from dump_cnn_features import save_features
import argparse

use_gpu = torch.cuda.is_available()
debugging = not use_gpu
should_dump = True#not debugging
should_covert_to_words = not debugging
should_dump_indices = not debugging


EPOCHS = 60 if not debugging else 3
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
BATCH_SIZE = 128 if not debugging else 8
K = 3  # number of distractors
n_image_features = 2048#4096

# Default settings
vocab_size = 10
max_sentence_length = 5
shapes_dataset = 'balanced_3_3'
vl_loss_weight = 0.0
bound_weight = 1.0
should_train_visual = False
cnn_model_file_name = None
rsa_sampling = 50
seed = 42
use_symbolic_input = False


cmd_parser = argparse.ArgumentParser()
cmd_parser.add_argument('seed', type=int)
cmd_parser.add_argument('vocab_size', type=int)
cmd_parser.add_argument('max_sentence_length', type=int)
cmd_parser.add_argument('vl_loss_weight', type=float)
cmd_parser.add_argument('bound_weight', type=float)
cmd_parser.add_argument('--shapes_dataset')
cmd_parser.add_argument('--use_symbolic_input', action='store_true')

excl_group = cmd_parser.add_mutually_exclusive_group()
excl_group.add_argument('--should_train_visual', action='store_true')
excl_group.add_argument('--cnn_model_file_name')

cmd_parser.add_argument('rsa_sampling', type=int)

cmd_args = cmd_parser.parse_args()

# Overwrite default settings if given in command line
if len(sys.argv) > 1:
	seed = cmd_args.seed #int(sys.argv[1])
	vocab_size = cmd_args.vocab_size #int(sys.argv[2])
	max_sentence_length = cmd_args.max_sentence_length #int(sys.argv[3])
	shapes_dataset = cmd_args.shapes_dataset #sys.argv[4]
	vl_loss_weight = cmd_args.vl_loss_weight #float(sys.argv[5])
	bound_weight = cmd_args.bound_weight #float(sys.argv[6])
	use_symbolic_input = cmd_args.use_symbolic_input
	should_train_visual = cmd_args.should_train_visual
	cnn_model_file_name = cmd_args.cnn_model_file_name
	rsa_sampling = cmd_args.rsa_sampling

# Symbolic input or mscoco never need to train visual features
if not use_symbolic_input and not shapes_dataset is None:
	assert should_train_visual or cnn_model_file_name is not None, 'Need stored CNN weights if not training visual features'

# Get model id using a timestamp
model_id = '{:%m%d%H%M%S%f}'.format(datetime.now())
starting_epoch = 0


################# Print info ####################
print('========================================')
print('Model id: {}'.format(model_id))
print('Seed: {}'.format(seed))	
print('Training visual module: {}'.format(should_train_visual))
if not should_train_visual and not use_symbolic_input and not shapes_dataset is None:
	print('Loading pretrained CNN from: {}'.format(cnn_model_file_name))
print('|V|: {}'.format(vocab_size))
print('L: {}'.format(max_sentence_length))
print('Using gpu: {}'.format(use_gpu))
if not shapes_dataset is None:
	print('Dataset: {} ({})'.format(shapes_dataset, 'symbolic' if use_symbolic_input else 'pixels'))
else:
	print('Dataset: mscoco')
print('Lambda: {}'.format(vl_loss_weight))
print('Alpha: {}'.format(bound_weight))
if not use_symbolic_input and not shapes_dataset is None:
	print('N image features: {}'.format(n_image_features))
if rsa_sampling >= 0:
	print('N samples for RSA: {}'.format(rsa_sampling))
print()
#################################################

if not shapes_dataset is None:
	# Create vocab if there is not one for the desired size already
	if not does_vocab_exist(vocab_size):
		build_vocab(vocab_size)

# Load vocab
word_to_idx, idx_to_word, bound_idx = load_dictionaries(
	'shapes' if not shapes_dataset is None else 'mscoco', 
	vocab_size)

# Load pretrained CNN if necessary
if not should_train_visual and not use_symbolic_input and not shapes_dataset is None:
	cnn_model_id = cnn_model_file_name.split('/')[-1]

	features_folder_name = 'data/shapes/{}_{}'.format(shapes_dataset, cnn_model_id)

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
		features_folder_name = save_features(trained_cnn, shapes_dataset, cnn_model_id)


if not shapes_dataset is None:
	# Create onehot metadata if not created yet
	if not does_shapes_onehot_metadata_exist(shapes_dataset):
		create_shapes_onehot_metadata(shapes_dataset)

	# Load metadata
	train_metadata, valid_metadata, test_metadata = load_shapes_onehot_metadata(shapes_dataset)
else:
	train_metadata = None
	valid_metadata = None
	test_metadata = None


# Load data
if not shapes_dataset is None:
	if not use_symbolic_input:
		if should_train_visual:
			train_data, valid_data, test_data = load_images('shapes/{}'.format(shapes_dataset), BATCH_SIZE, K)
		else:
			n_pretrained_image_features, train_data, valid_data, test_data = load_pretrained_features(
				features_folder_name, BATCH_SIZE, K)
			assert n_pretrained_image_features == n_image_features
	else:
		n_image_features, train_data, valid_data, test_data = load_pretrained_features(
			'shapes/{}'.format(shapes_dataset), BATCH_SIZE, K, use_symbolic=True)
else:
	n_image_features, train_data, valid_data, test_data = load_pretrained_features(
			'data/mscoco', BATCH_SIZE, K)
	print('\nUsing {} image features\n'.format(n_image_features))


# Settings
dumps_dir = './dumps'
if should_dump and not os.path.exists(dumps_dir):
	os.mkdir(dumps_dir)


current_model_dir = '{}/{}'.format(dumps_dir, model_id)

if should_dump and not os.path.exists(current_model_dir):
	os.mkdir(current_model_dir)


should_evaluate_best = True
# Evaluate best model on test data
if should_evaluate_best:

	if debugging:
		# Just pick the latest
		best_model = model
		best_epoch = e
	else:
		# Actually pick the best
		best_epoch = 52
		best_model = Model(n_image_features, vocab_size,
			EMBEDDING_DIM, HIDDEN_SIZE, 
			bound_idx, max_sentence_length, 
			vl_loss_weight, bound_weight, 
			should_train_visual, rsa_sampling,
			use_gpu)
		best_model_name = 'dumps/0510200906157042/0510200906157042_52_model'
		state = torch.load(best_model_name, map_location= lambda storage, location: storage)
		best_model.load_state_dict(state)

		print()
		print('Best model is in file: {}'.format(best_model_name))

	if use_gpu:
		best_model = best_model.cuda()

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
	test_language_entropy_meter) = evaluate(best_model, test_data, test_word_counts, test_metadata, debugging)

	print()
	print('Test accuracy: {}'.format(test_acc_meter.avg))

	if should_dump:
		pickle.dump(test_loss_meter, open('{}/{}_{}_test_losses_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
		pickle.dump(test_acc_meter, open('{}/{}_{}_test_accuracy_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
		pickle.dump(test_entropy_meter, open('{}/{}_{}_test_entropy_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
		pickle.dump(test_distinctness_meter, open('{}/{}_{}_test_distinctness_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
		pickle.dump(test_rsa_sr_meter, open('{}/{}_{}_test_rsa_sr_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
		pickle.dump(test_rsa_si_meter, open('{}/{}_{}_test_rsa_si_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
		pickle.dump(test_rsa_ri_meter, open('{}/{}_{}_test_rsa_ri_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
		pickle.dump(test_topological_sim_meter, open('{}/{}_{}_test_topological_sim_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
		pickle.dump(test_language_entropy_meter, open('{}/{}_{}_test_language_entropy_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
		pickle.dump(test_messages, open('{}/{}_{}_test_messages.p'.format(current_model_dir, model_id, best_epoch), 'wb'))

		if should_dump_indices:
			pickle.dump(test_indices, open('{}/{}_{}_test_imageIndices.p'.format(current_model_dir, model_id, best_epoch), 'wb'))			

		if should_covert_to_words:
			dump_words(current_model_dir, test_messages, idx_to_word, '{}_{}_test_messages'.format(model_id, best_epoch))