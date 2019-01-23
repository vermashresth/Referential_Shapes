import pickle
import numpy as np
import random

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from ImageDataset import ImageDataset, ImagesSampler

from utils import AverageMeter

EPOCHS = 2#1000
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
BATCH_SIZE = 128
MAX_SENTENCE_LENGTH = 5#13
START_TOKEN = '<S>'
K = 2 # number of distractors


class Sender(nn.Module):
	def __init__(self, n_image_features, vocab_size):
		super().__init__()

		self.lstm_cell = nn.LSTMCell(EMBEDDING_DIM, HIDDEN_SIZE)
		self.aff_transform = nn.Linear(n_image_features, HIDDEN_SIZE)
		self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
		self.linear_probs = nn.Linear(HIDDEN_SIZE, vocab_size) # from a hidden state to the vocab

	def forward(self, t, start_token_idx, greedy=True):
		message = torch.zeros([BATCH_SIZE, MAX_SENTENCE_LENGTH], dtype=torch.long)

		# h0, c0, w0
		h = self.aff_transform(t) # BATCH_SIZE, HIDDEN_SIZE
		c = torch.zeros([BATCH_SIZE, HIDDEN_SIZE])
		w = self.embedding(torch.ones([BATCH_SIZE], dtype=torch.long) * start_token_idx)

		for i in range(MAX_SENTENCE_LENGTH): # or sampled <S>, but this is batched
			h, c = self.lstm_cell(w, (h, c))

			p = F.softmax(self.linear_probs(h), dim=1)

			if self.training or not self.greedy:
				cat = Categorical(p)
				w_idx = cat.sample() # rsample?
			else:
				_, w_idx = torch.max(p, -1)

			message[:,i] = w_idx

			# For next iteration
			w = self.embedding(torch.LongTensor(w_idx))

		return message, cat.log_prob(w_idx)


class Receiver(nn.Module):
	def __init__(self, n_image_features, vocab_size):
		super().__init__()

		self.lstm_cell = nn.LSTMCell(EMBEDDING_DIM, HIDDEN_SIZE)
		self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
		self.aff_transform = nn.Linear(HIDDEN_SIZE, n_image_features)

	def forward(self, m):
		# h0, c0
		h = torch.zeros([BATCH_SIZE, HIDDEN_SIZE])
		c = torch.zeros([BATCH_SIZE, HIDDEN_SIZE])

		# Need to change to batch dim second to iterate over tokens in message
		m = m.permute(1, 0)

		for w_idx in m:
			emb = self.embedding(w_idx)
			h, c = self.lstm_cell(emb, (h, c))

		return self.aff_transform(h)


class Model(nn.Module):
	def __init__(self, n_image_features, vocab_size):
		super().__init__()

		self.sender = Sender(n_image_features, vocab_size)
		self.receiver = Receiver(n_image_features, vocab_size)

	def forward(self, target, distractors, word_to_idx):
		m, log_prob = self.sender(target, word_to_idx[START_TOKEN])

		r_transform = self.receiver(m) # g(.)

		loss = 0
		r_transform = r_transform.permute(1,0)

		for d in distractors:
			loss += torch.max(torch.tensor(0.0), 1.0 - target @ r_transform + d @ r_transform)

		loss = -loss * log_prob
		
		return torch.mean(loss)


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
model = Model(n_image_features, vocab_size)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train

losses_meters = []

for e in range(EPOCHS):

	### Remove #####
	counter = 0

	epoch_loss_meter = AverageMeter()

	for d in train_data:
		optimizer.zero_grad()

		target, distractors = d
	
		loss = model(target, distractors, word_to_idx)

		print(loss.item())
		epoch_loss_meter.update(loss.item())

		loss.backward()
		
		optimizer.step()
		
		##### REMOVE ######
		counter +=1
		if counter == 10:
			break

	losses_meters.append(epoch_loss_meter)

	print('Epoch {}, average loss: {}'.format(e, losses_meters[e].avg))





