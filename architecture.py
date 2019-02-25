# Senior Project: HAR using STPN
# Network Architecture
# Date: Feb. 25, 2019

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)


class STPN(torch.nn.Module):

	# Inception models exceptcs tensors with a size of N x 3 x 299 x 299

	def __init__(self):
		super(STPN, self).__init__()

		# Spatial Stream
		# Input channels = 3
		self.cnn1 = models.inception_v3(pretrained = True)
		# torch.nn.AvgPool2d(kernel_size, stride = None, padding = 0, ceil_mode = False, count_include_pad=True)
		# ceil_mode - when True, will use ceil instead of floor to compute the output shape
		# count_include_pad - when True, will include the zero-padding in the average calculation
		# Kernel size: 7 x 7
		self.pool1 = torch.nn.AvgPool2d((7, 7), stride, padding)

		# Temporal Stream
		# Input channels = 3
		self.cnn2 = models.inception_v3(pretrained = True)
		self.cnn3 = models.inception_v3(pretrained = True)
		self.cnn4 = models.inception_v3(pretrained = True)

		self.pool2 = torch.nn.AvgPool2D((7, 7), stride, padding)

		# Attention stream
		# STCB layers
		# self.stcb1 = CompactBilinearPooling()
		# self.stcb2 = CompactBilinearPooling()

		# Input channels = 7, output channels = 7
		self.conv1 = torch.nn.Conv2d(2048, 64, (7, 7), stride, padding)
		self.conv2 = torch.nn.Conv2d(64, 1, (7, 7), stride, padding)
		self.sm = torch.nn.Softmax2d()
		# self.pool3 = WeightedPool()

		# Intersection of streams
		# self.stcb3 = CompactBilinearPooling()
		# 4096 input features, 101 output features for 101 defined classes
		self.fc = torch.nn.Linear(4096, 101)

	def forward(self, x):
		None

def createLossAndOptimizer(net, learning_rate = 0.001):
	# Loss function
	loss = torch.nn.CrossEntropyLoss()

	# Optimizer
	optimizer = optim.Adam(net.parameters(), lr = learning_rate)

	return (loss, optimizer)