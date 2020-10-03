import pickle
import numpy as np
import random
from datetime import datetime
import os
import sys
import time
import math

import torch
from viz_grad_cam_model import Model
from run import train_one_epoch, evaluate
from utils import EarlyStopping
from dataloader import load_dictionaries, load_images, load_pretrained_features
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


EPOCHS = 10 if not debugging else 3
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
BATCH_SIZE = 128 if not debugging else 8
K = 3  # number of distractors
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
rsa_sampling = -1
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
cmd_parser.add_argument('--use_bullet', type=int, default=0)
cmd_parser.add_argument('--epochs', type=int, default=EPOCHS)
cmd_parser.add_argument('--use_symbolic_input', action='store_true', default=use_symbolic_input)
cmd_parser.add_argument('--use_distractors_in_sender', type=int, default=use_distractors_in_sender)

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
	print("Not Supported type")

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

if not should_train_visual:
		to_load_model_id = model_id.replace('pre', 'train')
		to_load_current_model_dir = current_model_dir.replace('pre', 'train')
		cnn_model_file_name = '{}/{}_{}_model'.format(to_load_current_model_dir, to_load_model_id, EPOCHS - 1)

starting_epoch = 0

wandb.init(project="referential-shapes", name=model_id)

wandb.config.K = K #int(sys.argv[1])
wandb.config.seed = seed #int(sys.argv[1])
wandb.config.vocab_size = vocab_size #int(sys.argv[2])
wandb.config.max_sentence_length = max_sentence_length #int(sys.argv[3])
wandb.config.dataset_name = dataset_name #sys.argv[4]
wandb.config.vl_loss_weight = vl_loss_weight #float(sys.argv[5])
wandb.config.bound_weight = bound_weight #float(sys.argv[6])
wandb.config.use_symbolic_input = use_symbolic_input
wandb.config.use_distractors_in_sender = use_distractors_in_sender
wandb.config.should_train_visual = should_train_visual
wandb.config.cnn_model_file_name = cnn_model_file_name
wandb.config.use_random_model = use_random_model
wandb.config.rsa_sampling = rsa_sampling
wandb.config.noise_strength = noise_strength
wandb.config.repr = repr
wandb.config.exp_id = model_id[6:]


# ################# Print info ####################
# print('========================================')
# print('Model id: {}'.format(model_id))
# print('Seed: {}'.format(seed))
# print('Training visual module: {}'.format(should_train_visual))
# if not should_train_visual and not use_symbolic_input and not shapes_dataset is None:
# 	print('Loading pretrained CNN from: {}'.format(cnn_model_file_name))
# print('|V|: {}'.format(vocab_size))
# print('L: {}'.format(max_sentence_length))
# print('Using gpu: {}'.format(use_gpu))
# if not shapes_dataset is None:
# 	print('Dataset: {} ({})'.format(shapes_dataset, 'symbolic' if use_symbolic_input else 'pixels'))
# else:
# 	print('Dataset: mscoco')
# print('Lambda: {}'.format(vl_loss_weight))
# print('Alpha: {}'.format(bound_weight))
# if not use_symbolic_input and not shapes_dataset is None:
# 	print('N image features: {}'.format(n_image_features))
# if rsa_sampling >= 0:
# 	print('N samples for RSA: {}'.format(rsa_sampling))
# print()
# #################################################

# print("build vocab")
if not shapes_dataset is None:
	# Create vocab if there is not one for the desired size already
	if not does_vocab_exist(vocab_size):
		build_vocab(vocab_size)

# print("loading vocab")
# Load vocab
word_to_idx, idx_to_word, bound_idx = load_dictionaries(
	'shapes' if not shapes_dataset is None else 'mscoco',
	vocab_size)
# print("loading pretrained cnn")
# Load pretrained CNN if necessary
# if not should_train_visual and not use_symbolic_input and not shapes_dataset is None:
# 	cnn_model_id = cnn_model_file_name.split('/')[-1]
#
# 	features_folder_name = 'data/shapes/{}_{}'.format(shapes_dataset, cnn_model_id)
#
# 	# Check if the features were already extracted with this CNN
# 	if not os.path.exists(features_folder_name):
# 		# Load CNN from dumped model
# 		state = torch.load(cnn_model_file_name, map_location= lambda storage, location: storage)
# 		cnn_state = {k[4:]:v for k,v in state.items() if 'cnn' in k}
# 		trained_cnn = CNN(n_image_features)
# 		trained_cnn.load_state_dict(cnn_state)
#
# 		if use_gpu:
# 			trained_cnn = trained_cnn.cuda()
#
# 		print("=CNN state loaded=")
# 		print("Extracting features...")
#
# 		# Dump the features to then load them
# 		features_folder_name = save_features(trained_cnn, shapes_dataset, cnn_model_id)

# print("crating one hot metadata")
if not shapes_dataset is None:
	# Create onehot metadata if not created yet
	if not does_shapes_onehot_metadata_exist(shapes_dataset):
		create_shapes_onehot_metadata(shapes_dataset)

	# Load metadata
	train_metadata, valid_metadata, test_metadata, noise_metadata = load_shapes_onehot_metadata(shapes_dataset)
else:
	train_metadata = None
	valid_metadata = None
	test_metadata = None
	noise_metadata = None
# print("loaded metadata")
# print("loading data")
# Load data
# if not shapes_dataset is None:
# 	if not use_symbolic_input:
# 		if should_train_visual:
# 			train_data, valid_data, test_data, noise_data = load_images('shapes/{}'.format(shapes_dataset), BATCH_SIZE, K)
# 		else:
# 			n_pretrained_image_features, train_data, valid_data, test_data, noise_data = load_pretrained_features(
# 				features_folder_name, BATCH_SIZE, K)
# 			assert n_pretrained_image_features == n_image_features
# 	else:
# 		n_image_features, train_data, valid_data, test_data, noise_data= load_pretrained_features(
# 			'shapes/{}'.format(shapes_dataset), BATCH_SIZE, K, use_symbolic=True)
# else:
# 	n_image_features, train_data, valid_data, test_data, noise_data = load_pretrained_features(
# 			'data/mscoco', BATCH_SIZE, K)
# 	print('\nUsing {} image features\n'.format(n_image_features))
#
# print("data loaded")
# Settings
should_train_visual = 1

print("creating model")
model = Model(n_image_features, vocab_size,
	EMBEDDING_DIM, HIDDEN_SIZE,
	bound_idx, max_sentence_length,
	vl_loss_weight, bound_weight,
	should_train_visual, rsa_sampling,
	use_gpu, K, use_distractors_in_sender)

# wandb.watch(model)
model.eval()
print("model created")
if use_gpu:
	model = model.cuda()
print("model moved to gpu")

state = torch.load(cnn_model_file_name, map_location= lambda storage, location: storage)
model.load_state_dict(state)

# print("Model's state_dict:")
# for param_tensor in model.state_dict():
#     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
# for name, layer in model.named_modules():
#   print(name, layer)

train_data, valid_data, test_data, noise_data = load_images('shapes/{}'.format(shapes_dataset), BATCH_SIZE, K)


word_counts = torch.zeros([vocab_size])

# (loss,
# 		acc,
# 		m,
#     scores,
# 		batch_w_counts,
# 		entropy,
# 		distinctness,
# 		rsa_sr,
# 		rsa_si,
# 		rsa_ri,
# 		topological_sim,
# 		posdis,
# 		bosdis,
# 		lang_entropy) = model(target, distractors, word_counts, None)
# print(m)
# print('base acc', acc)
# print(target.size())
# print(len(distractors))
# print(distractors[0].size())
# print(idx)

# print(target.size())
# print(distractors[0].size())

word_counts = torch.zeros([vocab_size])


from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
import torch.nn as nn
class simpleModel(nn.Module):
  def __init__(self, model, variable, word_counts, mode):
    super().__init__()
    self.model = model
    self.model.mode = mode
    if mode=='r_t' or mode=='s_t':
        print("case 0")
        self.distractors = variable
    else:
        print("case 1")
        self.target = variable
    self.word_counts = word_counts
  def forward(self, inp):
    # print(cat_target.size(), 'cat')
    # target, distractors = torch.split(cat_target, [30,30], -1)
    # distractors = [distractors]
    # print(target.size())
    # print(distractors[0].size())
    if self.model.mode in ['s_t', 'r_t']:
        (loss,
    		acc,
    		m,
        vocab_scores_tensor,
    		batch_w_counts,
    		entropy,
    		distinctness,
    		rsa_sr,
    		rsa_si,
    		rsa_ri,
    		topological_sim,
    		posdis,
    		bosdis,
    		lang_entropy) = self.model(inp, self.distractors, self.word_counts, None)
    else:
        (loss,
    		acc,
    		m,
        vocab_scores_tensor,
    		batch_w_counts,
    		entropy,
    		distinctness,
    		rsa_sr,
    		rsa_si,
    		rsa_ri,
    		topological_sim,
    		posdis,
    		bosdis,
    		lang_entropy) = self.model(self.target, [inp], self.word_counts, None)

    print('acc', acc)
    return vocab_scores_tensor
images = []
ct = 0
for batch in valid_data: # or anything else you want to do
  if ct==6:
    break
  target, distractors, idx = batch
  target = target[0].unsqueeze(0)
  distractors = [distractors[0][0].unsqueeze(0)]

  sm = simpleModel(model, distractors, word_counts, 's_t')
  sm.eval()
  gradcam = GradCAM(sm, sm.model.cnn.conv_net[8])
  mask, _ = gradcam(target)
  heatmap_s, result = visualize_cam(mask, target)
  model.zero_grad()

  sm = simpleModel(model, target, word_counts, 's_d')
  sm.eval()
  gradcam = GradCAM(sm, sm.model.cnn.conv_net[6])
  mask, _ = gradcam(distractors[0])
  heatmap_s_d, result = visualize_cam(mask, distractors[0])
  model.zero_grad()

  sm = simpleModel(model, distractors, word_counts, 'r_t')
  sm.train()
  gradcam = GradCAM(sm, sm.model.cnn.conv_net[6])
  mask, _ = gradcam(target)
  heatmap_r, result = visualize_cam(mask, target)
  model.zero_grad()

  sm = simpleModel(model, target, word_counts, 'r_d')
  sm.train()
  gradcam = GradCAM(sm, sm.model.cnn.conv_net[6])
  mask, _ = gradcam(distractors[0])
  heatmap_r_d, result = visualize_cam(mask, distractors[0])

  # print(valid_data.dataset.mean, valid_data.dataset.std)
  np.save('image2.npy', target.cpu().numpy()*valid_data.dataset.std[0]+valid_data.dataset.mean[0])
  np.save('distractor2.npy', distractors[0].cpu().numpy()*valid_data.dataset.std[0]+valid_data.dataset.mean[0])
  np.save('result2.npy', result)
  np.save('heatmap2_s.npy', heatmap_s)
  np.save('heatmap2_r.npy', heatmap_r)
  np.save('heatmap2_r_d.npy', heatmap_r_d)

  from torchvision.utils import make_grid
  from torchvision import transforms
  import matplotlib.pyplot as plt

  images.extend([torch.Tensor(target.cpu().numpy()[0]),
                torch.Tensor(distractors[0].cpu()[0].numpy()),
                torch.Tensor(heatmap_s),
				torch.Tensor(heatmap_s_d),
                torch.Tensor(heatmap_r),
                torch.Tensor(heatmap_r_d)])
  ct+=1
grid_image = make_grid(images, nrow=6)
img = grid_image.numpy()
np.save('grid.npy', img)
