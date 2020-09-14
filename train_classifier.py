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
from metadata import does_shapes_classdata_exist, create_shapes_classdata, load_shapes_classdata
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

# Symbolic input or mscoco never need to train visual features
# if not use_symbolic_input and not shapes_dataset is None:
# 	assert should_train_visual or cnn_model_file_name is not None, 'Need stored CNN weights if not training visual features'

# Get model id using a timestamp
repr = 'classifier'

model_id = 'seed-{}_K-{}_repr-{}_distractor-aware-{}_data-{}_noise-{}'.format(seed, K, repr, use_distractors_in_sender, dataset_name, noise_strength)

dumps_dir = './dumps'
if should_dump and not os.path.exists(dumps_dir):
	os.mkdir(dumps_dir)

current_model_dir = '{}/{}'.format(dumps_dir, model_id)

if should_dump and not os.path.exists(current_model_dir):
	os.mkdir(current_model_dir)


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
wandb.config.should_train_visual = should_train_visual
wandb.config.cnn_model_file_name = cnn_model_file_name
wandb.config.use_random_model = use_random_model
wandb.config.rsa_sampling = rsa_sampling
wandb.config.noise_strength = noise_strength
wandb.config.repr = repr
wandb.config.exp_id = model_id[6:]

################# Print info ####################
print('========================================')
print('Model id: {}'.format(model_id))
print('Seed: {}'.format(seed))
print('Using gpu: {}'.format(use_gpu))
if not shapes_dataset is None:
	print('Dataset: {} ({})'.format(shapes_dataset, 'symbolic' if use_symbolic_input else 'pixels'))
else:
	print('Dataset: mscoco')

#################################################


print("crating one hot metadata")

if not does_shapes_classdata_exist(shapes_dataset):
	create_shapes_classdata(shapes_dataset)

# Load metadata
train_metadata, valid_metadata, test_metadata, noise_metadata = load_shapes_classdata(shapes_dataset)

print("loaded metadata")
print("loading data")
# Load data
train_data, valid_data, test_data, noise_data = load_images('shapes/{}'.format(shapes_dataset), BATCH_SIZE, K)

print("data loaded")
# Settings

print("creating model")
cnnmodel = CNN(n_image_features)

import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, cnn, n_out_features, out_classes):
        super(MyModel, self).__init__()
        self.cnn = cnn
        self.fc = nn.Linear(n_out_features, out_classes)
    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        # x = nn.Softmax(x)
        return x

out_classes = 18
model = MyModel(cnnmodel, n_image_features, out_classes)

wandb.watch(model)

print("model created")
cnn_model_file_name = None
if only_eval:
    cnn_state = torch.load(cnn_model_file_name)
    cnn_state = {k[4:]:v for k,v in cnn_state.items() if 'cnn' in k}
    fc_state = torch.load('my_classifier_model')
    fc_state = {k:v for k,v in fc_state if 'fc' in k}
    model.cnn = model.cnn.load_state_dict(cnn_state)
    model.fc = model.fc.load_state_dict(fc_state)

# if use_gpu:
# 	model = model.cuda()
# print("model moved to gpu")


import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    running_acc = 0
    for i, data in enumerate(train_data) :
        # get the inputs; data is a list of [inputs, labels]
        if only_eval:
            model.eval()
        target, distractors, idx = data
        labels = torch.Tensor(np.array(train_metadata[idx[:,0]]).astype(int))
        labels = labels.type(torch.LongTensor)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(target)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        probs = nn.functional.softmax(outputs)
        _,pred = probs.max(-1)
        acc = sum(labels.detach().numpy()==pred.detach().numpy())/labels.size(0)
        # print statistics

        running_loss += loss.item()
        if i % 25 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f acc %.3f' %
                  (epoch + 1, i + 1, running_loss/25,acc ))
            running_loss = 0.0
            for val_data in valid_data:
                  target, distractors, idx = val_data
                  labels = torch.Tensor(np.array(valid_metadata[idx[:,0]]).astype(int))
                  labels = labels.type(torch.LongTensor)
                  model.eval()
                  # forward + backward + optimize
                  outputs = model(target)
                  probs = nn.functional.softmax(outputs)
                  _,pred = probs.max(-1)
                  acc = sum(labels.detach().numpy()==pred.detach().numpy())/labels.size(0)
                  print("eval acc ", acc)
                  break
            model.train()
torch.save(model.state_dict(), 'my_classifier_model')
