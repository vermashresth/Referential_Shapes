import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical


class Sender(nn.Module):
	def __init__(self, n_image_features, vocab_size, 
		embedding_dim, hidden_size, batch_size, use_gpu, greedy=True):
		super().__init__()

		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.greedy = greedy
		self.use_gpu = use_gpu
		self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_size)
		self.aff_transform = nn.Linear(n_image_features, hidden_size)
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.linear_probs = nn.Linear(hidden_size, vocab_size) # from a hidden state to the vocab

		self.reset_parameters()

	def reset_parameters(self):
		# nn.init.normal_(self.embd, 0.0, 0.1)

		nn.init.normal_(self.aff_transform.weight, 0, 0.1)
		nn.init.constant_(self.aff_transform.bias, 0)
		
		nn.init.constant_(self.linear_probs.weight, 0)
		nn.init.constant_(self.linear_probs.bias, 0)

		nn.init.xavier_uniform_(self.lstm_cell.weight_ih)
		nn.init.orthogonal_(self.lstm_cell.weight_hh)
		nn.init.constant_(self.lstm_cell.bias_ih, val=0)
       	# # cuDNN bias order: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
       	# # add some positive bias for the forget gates [b_i, b_f, b_o, b_g] = [0, 1, 0, 0]
		nn.init.constant_(self.lstm_cell.bias_hh, val=0)
		nn.init.constant_(self.lstm_cell.bias_hh[self.hidden_size:2 * self.hidden_size], val=1)

	def forward(self, t, start_token_idx, max_sentence_length):
		message = torch.zeros([self.batch_size, max_sentence_length], dtype=torch.long)

		# h0, c0, w0
		h = self.aff_transform(t) # batch_size, hidden_size
		c = torch.zeros([self.batch_size, self.hidden_size])
		
		idxs = torch.ones([self.batch_size], dtype=torch.long) * start_token_idx
		
		if self.use_gpu:
			message = message.cuda()
			c = c.cuda()
			idxs = idxs.cuda()

		w = self.embedding(idxs)

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
			w = self.embedding(w_idx)

		return message, cat.log_prob(w_idx)


class Receiver(nn.Module):
	def __init__(self, n_image_features, vocab_size,
		embedding_dim, hidden_size, batch_size, use_gpu):
		super().__init__()

		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.use_gpu = use_gpu
		self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_size)
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.aff_transform = nn.Linear(hidden_size, n_image_features)

		self.reset_parameters()

	def reset_parameters(self):
		# nn.init.normal_(self.embd, 0.0, 0.1)

		nn.init.normal_(self.aff_transform.weight, 0, 0.1)
		nn.init.constant_(self.aff_transform.bias, 0)

		nn.init.xavier_uniform_(self.lstm_cell.weight_ih)
		nn.init.orthogonal_(self.lstm_cell.weight_hh)
		nn.init.constant_(self.lstm_cell.bias_ih, val=0)
		# # cuDNN bias order: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
		# # add some positive bias for the forget gates [b_i, b_f, b_o, b_g] = [0, 1, 0, 0]
		nn.init.constant_(self.lstm_cell.bias_hh, val=0)
		nn.init.constant_(self.lstm_cell.bias_hh[self.hidden_size:2 * self.hidden_size], val=1)

	def forward(self, m):
		# h0, c0
		h = torch.zeros([self.batch_size, self.hidden_size])
		c = torch.zeros([self.batch_size, self.hidden_size])

		if self.use_gpu:
			h = h.cuda()
			c = c.cuda()

		# Need to change to batch dim second to iterate over tokens in message
		m = m.permute(1, 0)

		for w_idx in m:
			emb = self.embedding(w_idx)
			h, c = self.lstm_cell(emb, (h, c))

		return self.aff_transform(h)

class BaselineNN(nn.Module):
	def __init__(self, all_image_features, hidden_size):
		super().__init__()

		self.i2h = nn.Linear(all_image_features, hidden_size)
		self.h2h = nn.Linear(hidden_size, hidden_size)
		self.h2o = nn.Linear(hidden_size, 1)

	def forward(self, target, distractors):
		h = nn.Tanh()
		sigma = nn.Sigmoid()

		input = [target] # target + distractors
		for d in distractors:
			input.append(d)

		input = torch.cat(input, 1)

		return sigma(self.h2o(h(self.h2h(h(self.i2h(input)))))).squeeze()

class Model(nn.Module):
	def __init__(self, n_image_features, vocab_size,
		embedding_dim, hidden_size, batch_size, use_gpu):
		super().__init__()

		self.batch_size = batch_size
		self.use_gpu = use_gpu
		self.sender = Sender(n_image_features, vocab_size,
			embedding_dim, hidden_size, batch_size, use_gpu)
		self.receiver = Receiver(n_image_features, vocab_size,
			embedding_dim, hidden_size, batch_size, use_gpu)		

	def forward(self, target, distractors, word_to_idx, start_token, max_sentence_length, baseline_value):
		if self.use_gpu:
			target = target.cuda()
			distractors = [d.cuda() for d in distractors]

		m, log_prob = self.sender(target, word_to_idx[start_token], max_sentence_length)

		r_transform = self.receiver(m) # g(.)

		reward = 0

		target = target.view(self.batch_size, 1, -1)
		r_transform = r_transform.view(self.batch_size, -1, 1)

		target_score = torch.bmm(target, r_transform).squeeze() #scalar

		distractors_scores = []

		for d in distractors:
			d = d.view(self.batch_size, 1, -1)
			d_score = torch.bmm(d, r_transform).squeeze()
			distractors_scores.append(d_score)
			zero_tensor = torch.tensor(0.0)
			if self.use_gpu:
				zero_tensor = zero_tensor.cuda()

			reward += torch.max(zero_tensor, 1.0 - target_score + d_score)

		loss = (reward - baseline_value) * -log_prob
		
		loss = torch.mean(loss)

		# Calculate accuracy
		target_prob = torch.exp(target_score)

		all_probs = torch.zeros((self.batch_size, 1 + len(distractors)))
		all_probs[:,0] = target_prob

		for i, score in enumerate(distractors_scores):
			dist_prob = torch.exp(score)
			all_probs[:,i+1] = dist_prob

		_, max_idx = torch.max(all_probs, 1)

		accuracy = max_idx == 0
		accuracy = accuracy.to(dtype=torch.float32)
		accuracy = torch.mean(accuracy)

		return loss, accuracy, m, reward.data