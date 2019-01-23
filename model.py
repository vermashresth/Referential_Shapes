import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

class Sender(nn.Module):
	def __init__(self, n_image_features, vocab_size, 
		embedding_dim, hidden_size, batch_size, greedy=True):
		super().__init__()

		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.greedy = greedy
		self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_size)
		self.aff_transform = nn.Linear(n_image_features, hidden_size)
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.linear_probs = nn.Linear(hidden_size, vocab_size) # from a hidden state to the vocab

	def forward(self, t, start_token_idx, max_sentence_length):
		message = torch.zeros([self.batch_size, max_sentence_length], dtype=torch.long)

		# h0, c0, w0
		h = self.aff_transform(t) # batch_size, hidden_size
		c = torch.zeros([self.batch_size, self.hidden_size])
		w = self.embedding(torch.ones([self.batch_size], dtype=torch.long) * start_token_idx)

		for i in range(max_sentence_length): # or sampled <S>, but this is batched
			h, c = self.lstm_cell(w, (h, c))

			p = F.softmax(self.linear_probs(h), dim=1)

			cat = Categorical(p)

			if self.training or not self.greedy:	
				w_idx = cat.sample() # rsample?
			else:
				_, w_idx = torch.max(p, -1)

			message[:,i] = w_idx

			# For next iteration
			w = self.embedding(torch.LongTensor(w_idx))

		return message, cat.log_prob(w_idx)


class Receiver(nn.Module):
	def __init__(self, n_image_features, vocab_size,
		embedding_dim, hidden_size, batch_size):
		super().__init__()

		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_size)
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.aff_transform = nn.Linear(hidden_size, n_image_features)

	def forward(self, m):
		# h0, c0
		h = torch.zeros([self.batch_size, self.hidden_size])
		c = torch.zeros([self.batch_size, self.hidden_size])

		# Need to change to batch dim second to iterate over tokens in message
		m = m.permute(1, 0)

		for w_idx in m:
			emb = self.embedding(w_idx)
			h, c = self.lstm_cell(emb, (h, c))

		return self.aff_transform(h)


class Model(nn.Module):
	def __init__(self, n_image_features, vocab_size,
		embedding_dim, hidden_size, batch_size):
		super().__init__()

		self.sender = Sender(n_image_features, vocab_size,
			embedding_dim, hidden_size, batch_size)
		self.receiver = Receiver(n_image_features, vocab_size,
			embedding_dim, hidden_size, batch_size)

	def forward(self, target, distractors, word_to_idx, start_token, max_sentence_length):
		m, log_prob = self.sender(target, word_to_idx[start_token], max_sentence_length)

		r_transform = self.receiver(m) # g(.)

		loss = 0
		r_transform = r_transform.permute(1,0)

		for d in distractors:
			loss += torch.max(torch.tensor(0.0), 1.0 - target @ r_transform + d @ r_transform)

		loss = -loss * log_prob
		
		return torch.mean(loss)