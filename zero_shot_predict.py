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
from dataloader import load_dictionaries, load_images, load_pretrained_features_zero_shot
from build_shapes_dictionaries import *
from metadata import does_shapes_onehot_metadata_exist, create_shapes_onehot_metadata, load_shapes_onehot_metadata
from decode import dump_words
from visual_module import CNN
from dump_cnn_features import save_features
import argparse

import wandb

use_gpu = torch.cuda.is_available()
debugging = not use_gpu
should_dump = True#not debugging
should_covert_to_words = True#not debugging
should_dump_indices = True#not debugging


EPOCHS = 3 if not debugging else 3
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
BATCH_SIZE = 128 if not debugging else 8
K = 3  # number of distractors
n_image_features = 2048#4096

# Default settings
vocab_size = 10
max_sentence_length = 5
dataset_type = 4
pretrain_dataset_type = 0
vl_loss_weight = 0.0
bound_weight = 1.0
use_random_model = 0
should_train_visual = 0
cnn_model_file_name = None
rsa_sampling = 50
seed = 42
use_symbolic_input = False
noise_strength = 0
use_distractors_in_sender = 0

cmd_parser = argparse.ArgumentParser()
cmd_parser.add_argument('--K', type=int, default=K)
cmd_parser.add_argument('--seed', type=int, default=seed)
cmd_parser.add_argument('--vocab_size', type=int, default=vocab_size)
cmd_parser.add_argument('--max_sentence_length', type=int, default=max_sentence_length)
cmd_parser.add_argument('--vl_loss_weight', type=float, default=vl_loss_weight)
cmd_parser.add_argument('--bound_weight', type=float, default=bound_weight)
cmd_parser.add_argument('--noise_strength', type=int, default=noise_strength)
cmd_parser.add_argument('--dataset_type', type=int, default=dataset_type)
cmd_parser.add_argument('--pretrain_dataset_type', type=int, default=pretrain_dataset_type)
cmd_parser.add_argument('--use_symbolic_input', action='store_true', default=use_symbolic_input)
cmd_parser.add_argument('--use_distractors_in_sender', type=int, default=use_distractors_in_sender)
cmd_parser.add_argument('--use_bullet', type=int, default=0)
cmd_parser.add_argument('--epochs', type=int, default=EPOCHS)

cmd_parser.add_argument('--use_random_model', type=int, default=use_random_model)
cmd_parser.add_argument('--should_train_visual', type=int, default=should_train_visual)
# cmd_parser.add_argument('--cnn_model_file_name', type=str, default=cnn_model_file_name)

cmd_parser.add_argument('--rsa_sampling', type=int, default=rsa_sampling)

cmd_args = cmd_parser.parse_args()

# Overwrite default settings if given in command line
# if len(sys.argv) > 1:
K = cmd_args.K #int(sys.argv[1])
seed = cmd_args.seed #int(sys.argv[1])
vocab_size = cmd_args.vocab_size #int(sys.argv[2])
max_sentence_length = cmd_args.max_sentence_length #int(sys.argv[3])
dataset_type = cmd_args.dataset_type #sys.argv[4]
pretrain_dataset_type = cmd_args.pretrain_dataset_type #sys.argv[4]
vl_loss_weight = cmd_args.vl_loss_weight #float(sys.argv[5])
bound_weight = cmd_args.bound_weight #float(sys.argv[6])
use_symbolic_input = cmd_args.use_symbolic_input
should_train_visual = cmd_args.should_train_visual
use_random_model = cmd_args.use_random_model
rsa_sampling = cmd_args.rsa_sampling
noise_strength = cmd_args.noise_strength
use_distractors_in_sender = cmd_args.use_distractors_in_sender
use_bullet = cmd_args.use_bullet
EPOCHS = cmd_args.epochs

if dataset_type == 0: # Even, same pos
	shapes_dataset = 'get_dataset_balanced_incomplete_noise_{}_3_3'.format(noise_strength)
	dataset_name = 'even-samepos'
elif dataset_type == 1: # Even, diff pos
	shapes_dataset = 'get_dataset_different_targets_incomplete_noise_{}_3_3'.format(noise_strength)
	dataset_name = 'even-diffpos'
elif dataset_type == 2: # Uneven, same pos
	shapes_dataset = 'get_dataset_uneven_incomplete_noise_{}_3_3'.format(noise_strength)
	dataset_name = 'uneven-samepos'
elif dataset_type == 3: # Uneven,  diff pos
	shapes_dataset = 'get_dataset_uneven_different_targets_row_incomplete_noise_{}_3_3'.format(noise_strength)
	dataset_name = 'uneven-diffpos'
elif dataset_type == 4: #
	target_shapes_dataset = 'get_dataset_balanced_zero_shot_noise_{}_3_3'.format(noise_strength)
	distractors_shapes_dataset = 'get_dataset_balanced_zero_shot_noise_{}_3_3'.format(noise_strength)
	dataset_name = 'zero_shot'
	pretrain_dataset_name = ['even-samepos', 'even-diffpos', 'uneven-samepos', 'uneven-diffpos'][pretrain_dataset_type]
else:
	print("Not Supported type")
shapes_dataset = target_shapes_dataset
# Symbolic input or mscoco never need to train visual features
# if not use_symbolic_input and not shapes_dataset is None:
# 	assert should_train_visual or cnn_model_file_name is not None, 'Need stored CNN weights if not training visual features'

# Get model id using a timestamp
if should_train_visual:
	repr = 'train'
else:
	if use_random_model:
		repr = 'random'
	else:
		repr = 'pre'

model_id = 'seed-{}_K-{}_repr-{}_distractor-aware-{}_data-{}-bullet-{}_noise-{}'.format(seed, K, repr, use_distractors_in_sender, dataset_name, use_bullet, noise_strength)

dumps_dir = './dumps'
if should_dump and not os.path.exists(dumps_dir):
	os.mkdir(dumps_dir)

current_model_dir = '{}/{}'.format(dumps_dir, model_id)

if should_dump and not os.path.exists(current_model_dir):
	os.mkdir(current_model_dir)
learnt_random_model = None
if not should_train_visual:
	if use_random_model:
		cnn_model_file_name = './dumps/random/random_model'
		learnt_random_model = '{}/{}_{}_model'.format(current_model_dir,model_id, EPOCHS - 1)
		learnt_random_model = learnt_random_model.replace(dataset_name, pretrain_dataset_name)
		learnt_random_model = learnt_random_model.replace(dataset_name, pretrain_dataset_name)

	else:
		to_load_model_id = model_id.replace('pre', 'train')
		to_load_model_id = to_load_model_id.replace(dataset_name, pretrain_dataset_name)
		to_load_current_model_dir = current_model_dir.replace('pre', 'train')
		to_load_current_model_dir = to_load_current_model_dir.replace(dataset_name, pretrain_dataset_name)
		cnn_model_file_name = '{}/{}_{}_model'.format(to_load_current_model_dir, to_load_model_id, EPOCHS - 1)

starting_epoch = 0
if use_random_model:
	to_load_model_id = model_id
wandb.init(project="referential-shapes-clean", name=to_load_model_id)

wandb.config.K = K #int(sys.argv[1])
wandb.config.seed = seed #int(sys.argv[1])
wandb.config.vocab_size = vocab_size #int(sys.argv[2])
wandb.config.max_sentence_length = max_sentence_length #int(sys.argv[3])
wandb.config.dataset_name = dataset_name #sys.argv[4]
wandb.config.vl_loss_weight = vl_loss_weight #float(sys.argv[5])
wandb.config.bound_weight = bound_weight #float(sys.argv[6])
wandb.config.use_symbolic_input = use_symbolic_input
wandb.config.should_train_visual = should_train_visual
wandb.config.cnn_model_file_name = cnn_model_file_name
wandb.config.use_random_model = use_random_model
wandb.config.rsa_sampling = rsa_sampling
wandb.config.noise_strength = noise_strength
wandb.config.repr = repr
wandb.config.exp_id = to_load_model_id[6:]

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


# Create vocab if there is not one for the desired size already
if not does_vocab_exist(vocab_size):
	build_vocab(vocab_size)

# Load vocab
word_to_idx, idx_to_word, bound_idx = load_dictionaries('shapes', vocab_size)

model_file_name = cnn_model_file_name
# Load/generate features
cnn_dump_id = model_file_name.split('/')[-1]

target_features_folder_name = 'data/shapes/{}_{}'.format(target_shapes_dataset, cnn_dump_id)

# Check if the features were already extracted with this CNN
if not os.path.exists(target_features_folder_name):
	# Load CNN from dumped model
	state = torch.load(model_file_name, map_location= lambda storage, location: storage)
	cnn_state = {k[4:]:v for k,v in state.items() if 'cnn' in k}
	trained_cnn = CNN(n_image_features)
	trained_cnn.load_state_dict(cnn_state)

	if use_gpu:
		trained_cnn = trained_cnn.cuda()

	print("=CNN state loaded=")
	print("Extracting target features...")

	# Dump the features to then load them
	target_features_folder_name = save_features(trained_cnn, target_shapes_dataset, cnn_dump_id)

distractors_features_folder_name = 'data/shapes/{}_{}'.format(distractors_shapes_dataset, cnn_dump_id)

# Check if the features were already extracted with this CNN
if not os.path.exists(distractors_features_folder_name):
	# Load CNN from dumped model
	state = torch.load(model_file_name, map_location= lambda storage, location: storage)
	cnn_state = {k[4:]:v for k,v in state.items() if 'cnn' in k}
	trained_cnn = CNN(n_image_features)
	trained_cnn.load_state_dict(cnn_state)

	if use_gpu:
		trained_cnn = trained_cnn.cuda()

	print("=CNN state loaded=")
	print("Extracting distractors features...")

	# Dump the features to then load them
	distractors_features_folder_name = save_features(trained_cnn, target_shapes_dataset, cnn_dump_id)


# Load data
if should_train_visual:
	assert False
	_train_data, _valid_data, _test_data = load_images('shapes/{}'.format(target_shapes_dataset), BATCH_SIZE, K)
else:
	n_pretrained_image_features, _t, _v, test_data = load_pretrained_features_zero_shot(
		target_features_folder_name,
		distractors_features_folder_name,
		target_shapes_dataset,
		BATCH_SIZE,
		K)
	assert n_pretrained_image_features == n_image_features

# Create onehot metadata if not created yet - only target is needed
if not does_shapes_onehot_metadata_exist(target_shapes_dataset):
	create_shapes_onehot_metadata(target_shapes_dataset)

# Load metadata - only target is needed
_train_metadata, _valid_metadata, target_test_metadata, _ = load_shapes_onehot_metadata(target_shapes_dataset)


# Settings
dumps_dir = './dumps'
if should_dump and not os.path.exists(dumps_dir):
	os.mkdir(dumps_dir)

current_model_dir = '{}/{}_{}_{}'.format(dumps_dir, model_id, vocab_size, max_sentence_length)

if should_dump and not os.path.exists(current_model_dir):
	os.mkdir(current_model_dir)


model = Model(n_image_features, vocab_size,
	EMBEDDING_DIM, HIDDEN_SIZE,
	bound_idx, max_sentence_length,
	vl_loss_weight, bound_weight,
	should_train_visual, rsa_sampling,
	use_gpu, K, use_distractors_in_sender)

if learnt_random_model is not None:
  model_file_name = learnt_random_model
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
_pos,
_bos,
_) = evaluate(model, test_data, test_word_counts, target_test_metadata, debugging)

print()
print('Test accuracy: {}'.format(test_acc_meter.avg))
wandb.log({'Zero shot acc':test_acc_meter.avg})

# if should_dump:
# 	best_epoch = model_file_name.split('_')[-2]

# 	pickle.dump(test_loss_meter, open('{}/{}_{}_test_losses_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
# 	pickle.dump(test_acc_meter, open('{}/{}_{}_test_accuracy_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
# 	pickle.dump(test_entropy_meter, open('{}/{}_{}_test_entropy_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
# 	pickle.dump(test_distinctness_meter, open('{}/{}_{}_test_distinctness_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
# 	pickle.dump(test_rsa_sr_meter, open('{}/{}_{}_test_rsa_sr_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
# 	pickle.dump(test_rsa_si_meter, open('{}/{}_{}_test_rsa_si_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
# 	pickle.dump(test_rsa_ri_meter, open('{}/{}_{}_test_rsa_ri_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
# 	pickle.dump(test_topological_sim_meter, open('{}/{}_{}_test_topological_sim_meter.p'.format(current_model_dir, model_id, best_epoch), 'wb'))
# 	pickle.dump(test_messages, open('{}/{}_{}_test_messages.p'.format(current_model_dir, model_id, best_epoch), 'wb'))

# 	if should_dump_indices:
# 		pickle.dump(test_indices, open('{}/{}_{}_test_imageIndices.p'.format(current_model_dir, model_id, best_epoch), 'wb'))

# 	if should_covert_to_words:
# 		dump_words(current_model_dir, test_messages, idx_to_word, '{}_{}_test_messages'.format(model_id, best_epoch))
