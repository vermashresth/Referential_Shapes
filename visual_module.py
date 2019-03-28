import torch
import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):
	def __init__(self, use_gpu):
		super().__init__()

		

	def forward(self, x):
		

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






