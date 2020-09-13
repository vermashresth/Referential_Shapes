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
should_covert_to_words = True#not debugging
should_dump_indices = True#not debugging


EPOCHS = 60 if not debugging else 3
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
BATCH_SIZE = 128 if not debugging else 8
K = 1  # number of distractors
n_image_features = 2048#4096

# Default settings
vocab_size = 10
max_sentence_length = 5
dataset_type = 0
vl_loss_weight = 0.0
bound_weight = 1.0
use_random_model = 0
should_train_visual = 1
cnn_model_file_name = None
rsa_sampling = 50
seed = 42
use_symbolic_input = False
noise_strength = 0
use_distractors_in_sender = False

cmd_parser = argparse.ArgumentParser()
cmd_parser.add_argument('--K', type=int, default=K)
cmd_parser.add_argument('--seed', type=int, default=seed)
cmd_parser.add_argument('--vocab_size', type=int, default=vocab_size)
cmd_parser.add_argument('--max_sentence_length', type=int, default=max_sentence_length)
cmd_parser.add_argument('--vl_loss_weight', type=float, default=vl_loss_weight)
cmd_parser.add_argument('--bound_weight', type=float, default=bound_weight)
cmd_parser.add_argument('--noise_strength', type=int, default=noise_strength)
cmd_parser.add_argument('--dataset_type', type=int, default=dataset_type)
cmd_parser.add_argument('--use_symbolic_input', action='store_true', default=use_symbolic_input)
cmd_parser.add_argument('--use_distractors_in_sender', action='store_true', default=use_distractors_in_sender)

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
vl_loss_weight = cmd_args.vl_loss_weight #float(sys.argv[5])
bound_weight = cmd_args.bound_weight #float(sys.argv[6])
use_symbolic_input = cmd_args.use_symbolic_input
should_train_visual = cmd_args.should_train_visual
use_random_model = cmd_args.use_random_model
rsa_sampling = cmd_args.rsa_sampling
noise_strength = cmd_args.noise_strength
use_distractors_in_sender = cmd_args.use_distractors_in_sender

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
	print("Not Supported type")

if not shapes_dataset is None:
	# Create vocab if there is not one for the desired size already
	if not does_vocab_exist(vocab_size):
		build_vocab(vocab_size)

print("loading vocab")
# Load vocab
word_to_idx, idx_to_word, bound_idx = load_dictionaries(
	'shapes' if not shapes_dataset is None else 'mscoco',
	vocab_size)

model = Model(n_image_features, vocab_size,
	EMBEDDING_DIM, HIDDEN_SIZE,
	bound_idx, max_sentence_length,
	vl_loss_weight, bound_weight,
	should_train_visual, rsa_sampling,
	use_gpu, K, use_distractors_in_sender)

model_id = 'random'
dumps_dir = './dumps'
if should_dump and not os.path.exists(dumps_dir):
	os.mkdir(dumps_dir)
current_model_dir = '{}/{}'.format(dumps_dir, model_id)
if should_dump and not os.path.exists(current_model_dir):
	os.mkdir(current_model_dir)

torch.save(model.state_dict(), '{}/{}_model'.format(current_model_dir, model_id))
print("Random Model saved at {}".format('{}/{}_model'.format(current_model_dir, model_id)))
