import pickle
import numpy as np
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from ImageDataset import ImageDataset, ImagesSampler
from model import Sender, Receiver, Model
from run import train_one_epoch, evaluate

EPOCHS = 2#1000
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
BATCH_SIZE = 128
MAX_SENTENCE_LENGTH = 5#13
START_TOKEN = '<S>'
K = 2 # number of distractors


# Load data
with open("data/mscoco/dict.pckl", "rb") as f:
    d = pickle.load(f)
    word_to_idx = d["word_to_idx"] #dictionary w->i
    idx_to_word = d["idx_to_word"] #list of words
    bound_idx = word_to_idx["<S>"]

train_features = np.load('data/mscoco/train_features.npy')
valid_features = np.load('data/mscoco/valid_features.npy')
# 2d arrays of 4096 features

vocab_size = len(word_to_idx) # 10000
n_image_features = valid_features.shape[1] # 4096

train_dataset = ImageDataset(train_features)
valid_dataset = ImageDataset(valid_features, mean=train_dataset.mean, std=train_dataset.std) # All features are normalized with mean and std

train_data = DataLoader(train_dataset, num_workers=8, pin_memory=True, 
	batch_sampler=BatchSampler(ImagesSampler(train_dataset, K, shuffle=True), batch_size=BATCH_SIZE, drop_last=False))

valid_data = DataLoader(valid_dataset, num_workers=8, pin_memory=True,
	batch_sampler=BatchSampler(ImagesSampler(valid_dataset, K, shuffle=False), batch_size=BATCH_SIZE, drop_last=False))


# Settings
model = Model(n_image_features, vocab_size,
	EMBEDDING_DIM, HIDDEN_SIZE, BATCH_SIZE)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train

losses_meters = []
eval_losses_meters = []

for e in range(EPOCHS):
	epoch_loss_meter = train_one_epoch(model, train_data, optimizer, word_to_idx, START_TOKEN, MAX_SENTENCE_LENGTH)
	losses_meters.append(epoch_loss_meter)

	eval_loss_meter = evaluate(model, valid_data, word_to_idx, START_TOKEN, MAX_SENTENCE_LENGTH)
	eval_losses_meters.append(eval_loss_meter)

	print('Epoch {}, average train loss: {}, average val loss: {}'.format(
		e, losses_meters[e].avg, eval_losses_meters[e].avg))
