import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
#from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.categorical import Categorical

from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler

from ImageDataset import ImageDataset, ImagesSampler

EPOCHS = 1#1000
EMBEDDING_DIM = 256
HIDDEN_SIZE = 512
#VOCAB_SIZE = 10000 This comes from the data
BATCH_SIZE = 128
MAX_SENTENCE_LENGTH = 5#13
START_TOKEN = '<S>'
#TEMPERATURE = 1.2 only for ST-GS
#N_IMAGES_PER_ROUND = BATCH_SIZE# 127 distractors + 1 target
K = 2 # number of distractors


class Sender(nn.Module):
	def __init__(self, n_image_features, vocab_size):
		super().__init__()

		self.lstm_cell = nn.LSTMCell(EMBEDDING_DIM, HIDDEN_SIZE)
		self.aff_transform = nn.Linear(n_image_features, HIDDEN_SIZE)
		self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
		self.linear_probs = nn.Linear(HIDDEN_SIZE, vocab_size) # from a hidden state to the vocab

	def forward(self, t, start_token_idx):
		message = torch.zeros([BATCH_SIZE, MAX_SENTENCE_LENGTH], dtype=torch.long)

		# h0, c0, w0
		h = self.aff_transform(t) # BATCH_SIZE, HIDDEN_SIZE
		c = torch.zeros([BATCH_SIZE, HIDDEN_SIZE])
		w = self.embedding(torch.ones([BATCH_SIZE], dtype=torch.long) * start_token_idx)

		for i in range(MAX_SENTENCE_LENGTH): # or sampled <S>, but this is batched
			h, c = self.lstm_cell(w, (h, c))

			p = F.softmax(self.linear_probs(h), dim=1)

			cat = Categorical(p)
			w_idx = cat.sample() # rsample?

			message[:,i] = w_idx

			# For next iteration
			w = self.embedding(torch.LongTensor(w_idx))

		return message # batch size x L


class Receiver(nn.Module):
	def __init__(self, n_image_features, vocab_size):
		super().__init__()

		self.lstm_cell = nn.LSTMCell(EMBEDDING_DIM, HIDDEN_SIZE)
		self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)
		self.aff_transform = nn.Linear(HIDDEN_SIZE, n_image_features)#N_IMAGES_PER_ROUND why not?

	def forward(self, m, images):
		# h0, c0
		h = torch.zeros([BATCH_SIZE, HIDDEN_SIZE])
		c = torch.zeros([BATCH_SIZE, HIDDEN_SIZE])

		# Need to change to batch dim second to iterate over tokens in message
		m = m.permute(1, 0)

		for w_idx in m:
			emb = self.embedding(w_idx.long())
			h, c = self.lstm_cell(emb, (h, c))

		return self.aff_transform(h)


class Model(nn.Module):
	def __init__(self, n_image_features, vocab_size):
		super().__init__()

		self.sender = Sender(n_image_features, vocab_size)
		self.receiver = Receiver(n_image_features, vocab_size)

	def forward(self, target, distractors, word_to_idx):#images, word_to_idx):
		m = self.sender(target, word_to_idx[START_TOKEN])

		print(m.shape)

		assert False, 'just stop here!'

		images = None # target + distractors
		r_transform = self.receiver(m, images)

		print(r_transform.shape)

		#_, predictions = r_transform.max(1)

		# loss = torch.max(0, 1.0 - images @ r_transform + ? @ r_transform)

		assert False
		
		return 0 #loss


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

counter = 0

for _ in range(EPOCHS):
	for d in train_data:
		optimizer.zero_grad()

		target, distractors = d
	
		loss = model(target, distractors, word_to_idx)

		# loss.backward()
		
		# optimizer.step()
		
		##### REMOVE ######
		break





