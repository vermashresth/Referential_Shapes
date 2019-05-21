import torch
import torch.nn as nn
from torch.nn import functional as F

use_gpu=torch.cuda.is_available()
hidden_size = 64 # Check on this
embedding_dim = 32
#n_input_features = 11 # Max message length
vocab_size = 25

class FeatureRNN(nn.Module):
	def __init__(self, n_output_features):
		super().__init__()

		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)
		self.fc = nn.Linear(hidden_size, n_output_features)

	def forward(self, message, one_hot_expected):
		if use_gpu:
			message = message.cuda()
			one_hot_expected = one_hot_expected.cuda()

		emb = self.embedding(message)
		_out, (h,_c) = self.lstm(emb)

		h = h.squeeze()

		y = self.fc(h)
		#probs = F.softmax(y, dim=1) Do not do softmax caus CE does it

		ce_loss = nn.CrossEntropyLoss()

		_, labels = torch.max(one_hot_expected, 1)

		loss = ce_loss(y, labels)
		
		if self.training:
			return loss	
		else:
			# Calculate accuracy
			_, max_idx = torch.max(y, 1) # Do I not need probs here? I checked and it was the same
			acc = max_idx == labels
			acc = acc.to(dtype=torch.float32)

			return loss, torch.mean(acc)

