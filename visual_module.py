import torch
import torch.nn as nn
import torchvision.models as models

#Obverter's

# class CNNObverterLayer(nn.Module):
# 	def __init__(self, stride, use_gpu):
# 		super().__init__()

# 		self.conv = nn.Conv2D(in_channels=128, out_channels, kernel_size=3, stride=stride, padding=0)
# 		self.relu = nn.relu()

# 	def forward(self, x):

class CNN(nn.Module):
	def __init__(self, n_out_features):
		super().__init__()

		n_filters = 20
		self.conv_net = nn.Sequential(
			nn.Conv2d(3, n_filters, 3, stride=2),
			nn.BatchNorm2d(n_filters),
			nn.ReLU(),
			nn.Conv2d(n_filters, n_filters, 3, stride=1),
			nn.BatchNorm2d(n_filters),
			nn.ReLU(),
			nn.Conv2d(n_filters, n_filters, 3, stride=2),
			nn.BatchNorm2d(n_filters),
			nn.ReLU(),
			# nn.Conv2d(n_filters, n_filters, 3, stride=1),
			# nn.BatchNorm2d(n_filters),
			# nn.ReLU(),
			# nn.Conv2d(n_filters, n_filters, 3, stride=2),
			# nn.BatchNorm2d(n_filters),
			# nn.ReLU(),
			)

		self.lin = nn.Sequential(
			nn.Linear(500, n_out_features),
			nn.ReLU(),
			)

		self._init_params()

	def _init_params(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)


	def forward(self, x):
		batch_size = x.size(0)
		output = self.conv_net(x)
		output = output.view(batch_size, -1)
		output = self.lin(output)
		return output







# vgg16
# class CNN(nn.Module):
# 	def __init__(self, use_gpu):
# 		super().__init__()

# 		self.vgg16 = models.vgg16(pretrained=False)
# 		if use_gpu:
# 			self.vgg16 = self.vgg16.cuda()

# 	def forward(self, x):
# 		y = self.vgg16.features(x)
# 		y = y.view(y.size(0), -1)
# 		y = self.vgg16.classifier[:5](y)
# 		return y
