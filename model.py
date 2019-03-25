import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Sender(nn.Module):
	def __init__(self, n_image_features, vocab_size, 
		embedding_dim, hidden_size, batch_size, 
		bound_idx, max_sentence_length, vl_loss_weight, bound_weight,
		use_gpu, greedy=True):
		super().__init__()

		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.bound_token_idx = bound_idx
		self.max_sentence_length = max_sentence_length
		self.vl_loss_weight = vl_loss_weight
		self.bound_weight = bound_weight
		self.greedy = greedy
		self.use_gpu = use_gpu

		self.lstm_cell = nn.LSTMCell(embedding_dim, hidden_size)
		self.aff_transform = nn.Linear(n_image_features, hidden_size)
		self.embedding = nn.Parameter(torch.empty((vocab_size, embedding_dim), dtype=torch.float32))
		self.linear_probs = nn.Linear(hidden_size, vocab_size) # from a hidden state to the vocab

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.normal_(self.embedding, 0.0, 0.1)

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

	def _calculate_seq_len(self, seq_lengths, token, initial_length, seq_pos):	
		if self.training:
			max_predicted, vocab_index = torch.max(token, dim=1)
			mask = (vocab_index == self.bound_token_idx) * (max_predicted == 1.0)
		else:
			mask = token == self.bound_token_idx

		mask *= seq_lengths == initial_length
		seq_lengths[mask.nonzero()] = seq_pos + 1 # start symbol always appended

	def _discretize_token(self, token):
		if self.training:
			_, idx = torch.max(token, dim=1)
			return idx
		else:
			return token

	def forward(self, t, word_counts, tau=1.2):
		if self.training:
			message = [torch.zeros((self.batch_size, self.vocab_size), dtype=torch.float32)]
			if self.use_gpu:
				message[0] = message[0].cuda()
			message[0][:, self.bound_token_idx] = 1.0
		else:
			message = [torch.full((self.batch_size, ), fill_value=self.bound_token_idx, dtype=torch.int64)]
			if self.use_gpu:
				message[0] = message[0].cuda()

		# h0, c0
		h = self.aff_transform(t) # batch_size, hidden_size
		c = torch.zeros([self.batch_size, self.hidden_size])

		initial_length = self.max_sentence_length + 1
		seq_lengths = torch.ones([self.batch_size], dtype=torch.int64) * initial_length

		ce_loss = nn.CrossEntropyLoss(reduction='none')

		# Handle alpha for giving weight to the padding token
		bound_counts = word_counts[self.bound_token_idx]
		bound_counts *= self.bound_weight

		# word_counts[self.bound_token_idx] *= self.bound_weight # Don't do this because it's passed by reference

		assert self.bound_token_idx == len(word_counts) - 1, 'Bound token is not the last word in vocab'

		denominator = word_counts[:-1].sum() + bound_counts
		if denominator > 0:
			normalized_word_counts = word_counts / denominator
		else:
			normalized_word_counts = word_counts

		vl_loss = 0.0
		
		if self.use_gpu:
			c = c.cuda()
			seq_lengths = seq_lengths.cuda()

		for i in range(self.max_sentence_length): # or sampled <EOS>, but this is batched
			emb = torch.matmul(message[-1], self.embedding) if self.training else self.embedding[message[-1]]
			h, c = self.lstm_cell(emb, (h, c))

			vocab_scores = self.linear_probs(h)
			p = F.softmax(vocab_scores, dim=1)

			if self.training:	
				rohc = RelaxedOneHotCategorical(tau, p)
				token = rohc.rsample()

				# Straight-through part
				token_hard = torch.zeros_like(token)
				token_hard.scatter_(-1, torch.argmax(token, dim=-1, keepdim=True), 1.0)
				token = (token_hard - token).detach() + token
			else:
				if self.greedy:
					_, token = torch.max(p, -1)
				else:
					token = Categorical(p).sample()

			message.append(token)

			self._calculate_seq_len(seq_lengths, token, 
				initial_length, seq_pos=i+1)

			if self.vl_loss_weight > 0.0:
				vl_loss += ce_loss(vocab_scores - normalized_word_counts, self._discretize_token(token))

		return (torch.stack(message, dim=1), seq_lengths, vl_loss)


class Receiver(nn.Module):
	def __init__(self, n_image_features, vocab_size,
		embedding_dim, hidden_size, batch_size, use_gpu):
		super().__init__()

		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.use_gpu = use_gpu

		self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)
		self.embedding = nn.Parameter(torch.empty((vocab_size, embedding_dim), dtype=torch.float32))
		self.aff_transform = nn.Linear(hidden_size, n_image_features)

		self.reset_parameters()

	def reset_parameters(self):
		nn.init.normal_(self.embedding, 0.0, 0.1)

		nn.init.normal_(self.aff_transform.weight, 0, 0.1)
		nn.init.constant_(self.aff_transform.bias, 0)

		nn.init.xavier_uniform_(self.lstm.weight_ih_l0)
		nn.init.orthogonal_(self.lstm.weight_hh_l0)
		nn.init.constant_(self.lstm.bias_ih_l0, val=0)
		# cuDNN bias order: https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html#cudnnRNNMode_t
		# add some positive bias for the forget gates [b_i, b_f, b_o, b_g] = [0, 1, 0, 0]
		nn.init.constant_(self.lstm.bias_hh_l0, val=0)
		nn.init.constant_(self.lstm.bias_hh_l0[self.hidden_size:2 * self.hidden_size], val=1)

	def forward(self, m):
		# h0, c0
		h = torch.zeros([self.batch_size, self.hidden_size])
		c = torch.zeros([self.batch_size, self.hidden_size])

		if self.use_gpu:
			h = h.cuda()
			c = c.cuda()		

		emb = torch.matmul(m, self.embedding) if self.training else self.embedding[m]
		_, (h, c) = self.lstm(emb, (h[None, ...], c[None, ...]))

		return self.aff_transform(h)


class Model(nn.Module):
	def __init__(self, n_image_features, vocab_size,
		embedding_dim, hidden_size, batch_size, 
		bound_idx, max_sentence_length, vl_loss_weight, bound_weight, use_gpu):
		super().__init__()

		self.batch_size = batch_size
		self.use_gpu = use_gpu
		self.bound_token_idx = bound_idx
		self.max_sentence_length = max_sentence_length
		self.vocab_size = vocab_size
		self.vl_loss_weight = vl_loss_weight #lambda
		self.bound_weight = bound_weight #alpha

		self.sender = Sender(n_image_features, vocab_size,
			embedding_dim, hidden_size, batch_size, 
			bound_idx, max_sentence_length, vl_loss_weight, bound_weight, use_gpu)
		self.receiver = Receiver(n_image_features, vocab_size,
			embedding_dim, hidden_size, batch_size, use_gpu)

	def _pad(self, m, seq_lengths):
		max_len = m.shape[1]

		for e_i in range(self.batch_size):
			for i in range(seq_lengths[e_i], max_len):
				if self.training:
					m[e_i][i] = torch.zeros([1, self.vocab_size], dtype=torch.float32)
					m[e_i][i][self.bound_token_idx] = 1.0
					if self.use_gpu:
						m[e_i][i] = m[e_i][i].cuda()
				else:
					m[e_i][i] = self.bound_token_idx

		return m

	def _get_word_counts(self, m):
		if self.training:
			c = m.sum(dim=1).sum(dim=0).detach() # ToDo: are we sure about this???? Yeah, we are
		else:
			c = torch.zeros([self.vocab_size])
			if self.use_gpu:
				c = c.cuda()

			for w_idx in range(self.vocab_size):
				c[w_idx] = (m == w_idx).sum()
		return c

	def forward(self, target, distractors, word_counts):
		if self.use_gpu:
			target = target.cuda()
			distractors = [d.cuda() for d in distractors]

		use_different_targets = len(target.shape) == 3
		assert not use_different_targets or target.shape[1] == 2, 'This should only be two targets'

		if not use_different_targets:
			target_sender = target
			target_receiver = target
		else:
			target_sender = target[:, 0, :]
			target_receiver = target[:, 1, :]


		# Forward pass on Sender with its target
		m, seq_lengths, vl_loss = self.sender(target_sender, word_counts)

		# Pad with EOS tokens if EOS is predicted before max sentence length
		m = self._pad(m, seq_lengths)

		w_counts = self._get_word_counts(m)

		# Forward pass on Receiver with the message
		r_transform = self.receiver(m) # g(.)


		# Loss calculation
		loss = 0

		target_receiver = target_receiver.view(self.batch_size, 1, -1)
		r_transform = r_transform.view(self.batch_size, -1, 1)

		target_score = torch.bmm(target_receiver, r_transform).squeeze() #scalar

		distractors_scores = []

		for d in distractors:
			if use_different_targets:
				d = d[:, 0, :] # Just use the first distractor
			
			d = d.view(self.batch_size, 1, -1)
			d_score = torch.bmm(d, r_transform).squeeze()
			distractors_scores.append(d_score)
			zero_tensor = torch.tensor(0.0)
			if self.use_gpu:
				zero_tensor = zero_tensor.cuda()

			loss += torch.max(zero_tensor, 1.0 - target_score + d_score)

		# Calculate accuracy
		all_scores = torch.zeros((self.batch_size, 1 + len(distractors)))
		all_scores[:,0] = target_score

		for i, score in enumerate(distractors_scores):
			all_scores[:,i+1] = score

		all_scores = torch.exp(all_scores)

		_, max_idx = torch.max(all_scores, 1)

		accuracy = max_idx == 0 # target is the first element
		accuracy = accuracy.to(dtype=torch.float32)

		loss = loss + self.vl_loss_weight * vl_loss

		return torch.mean(loss), torch.mean(accuracy), m, w_counts




# max_len = m.shape[1] # self.max_sentence_length+1
		
		# print(seq_lengths)

		# # rows = torch.tensor()

		# # i = torch.LongTensor([[0],
		# # 					  [5]])
		# # v = torch.ones([self.batch_size*max_len - seq_lengths.sum()], dtype=torch.int64)
		# # mask = torch.sparse.LongTensor(i, v, torch.Size([self.batch_size, max_len])).to_dense()
		
		# # mask = (m[:,1:] == self.bound_token_idx)

		# mask = m == self.bound_token_idx
		
		# indices = mask.nonzero()

		# print(indices)
		
		# print(m)