import torch
import torch.nn as nn
from torch.nn import functional as F

use_gpu=torch.cuda.is_available()
hidden_size = 64 # Check on this
embedding_dim = 32
#n_input_features = 11 # Max message length
vocab_size = 25

class RNN(nn.Module):
	def __init__(self):
		super().__init__()

		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=1, batch_first=True)
		self.fc = nn.Linear(hidden_size, 3)

	def forward(self, message):
		emb = self.embedding(message)
		_out, (h,_c) = self.lstm(emb)

		h = h.squeeze()

		y = self.fc(h)
		#probs = F.softmax(y, dim=1) Do not do softmax caus CE does it

		return y


class Autoencoder(nn.Module):
	def __init__(self):
		super().__init__()
		self.encoder = RNN()
		self.decoder = nn.Sequential(
			nn.Linear(3, 12),
			nn.ReLU(True),
			nn.Linear(12, 64),
			nn.ReLU(True),
			nn.Linear(64, 128),
			nn.ReLU(True), nn.Linear(128, 30 * 30 * 3), nn.Tanh())

		# self.decoder = nn.Sequential(
		# 	nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
		# 	nn.ReLU(True),
		# 	nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
		# 	nn.ReLU(True),
		# 	nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
		# 	nn.Tanh()
		# 	)

	def forward(self, message, orig_image):
		if use_gpu:
			message = message.cuda()
			orig_image = orig_image.cuda()

		x = self.encoder(message)
		output = self.decoder(x)

		mse_loss = nn.MSELoss()

		orig_image = orig_image.view(orig_image.size(0), -1)

		loss = mse_loss(output, orig_image)

		return output, loss