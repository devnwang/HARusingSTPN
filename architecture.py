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

		# CNNs
		self.cnn1 = models.inception_v3(pretrained = True)
		self.cnn2 = models.inception_v3(pretrained = True)
		self.cnn3 = models.inception_v3(pretrained = True)
		self.cnn4 = models.inception_v3(pretrained = True)

		# Freeze the layers
		for param in self.cnn1.parameters, self.cnn2.parameters(), self.cnn3.parameters, self.cnn4.parameters():
			param.requires_grad = False

		# Construct last layer of CNNs
		num_ftrs1 = self.cnn1.fc.in_features
		self.cnn1.fc = torch.nn.Linear(num_ftrs1, 1024)

		num_ftrs2 = self.cnn2.fc.in_features
		self.cnn2.fc = torch.nn.Linear(num_ftrs2, 1024)

		num_ftrs3 = self.cnn2.fc.in_features
		self.cnn3.fc = torch.nn.Linear(num_ftrs3, 1024)

		num_ftrs4 = self.cnn2.fc.in_features
		self.cnn4.fc = torch.nn.Linear(num_ftrs4, 1024)


		# Spatial Stream
		# Input channels = 3
		#self.cnn1 = models.inception_v3(pretrained = True)
		# torch.nn.AvgPool2d(kernel_size, stride = None, padding = 0, ceil_mode = False, count_include_pad=True)
		# count_include_pad - when True, will include the zero-padding in the average calculation
		# Kernel size: 7 x 7
		self.pool1 = torch.nn.AvgPool2d((7, 7), stride, padding)

		# Temporal Stream
		# Input channels = 3
		#self.cnn2 = models.inception_v3(pretrained = True)
		#self.cnn3 = models.inception_v3(pretrained = True)
		#self.cnn4 = models.inception_v3(pretrained = True)

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
		# Substitute pooling layer
		self.pool3 = torch.nn.AvgPool2D((7, 7), stride, padding)

		# Intersection of streams
		# self.stcb3 = CompactBilinearPooling()
		# 4096 input features, 101 output features for 101 defined classes
		self.fc = torch.nn.Linear(4096, 101)

	def forward(self, rgb1, of1, of2, of3):
		# Spatial Stream
		rgb = self.cnn1(rgb1)
		spat = self.pool1(rgb)

		# Temporal Stream
		of1 = self.pool1(of1)
		of2 = self.pool1(of2)
		of3 = self.pool1(of3)
		# of = self.stcb1()
		temp = self.pool2(of)

		# Attention Stream
		#att1 = self.stcb2()
		att1 = self.conv1(att1)
		att1 = self.conv2(att1)
		att1 = self.sm(att1)
		att = self.pool3(att1)
		att = self.pool3(rgb)

		# spatemp = self.stcb3()

		res = self.fc(spatemp)

		

def createLossAndOptimizer(net, learning_rate = 0.001):
	# Loss function
	loss = torch.nn.CrossEntropyLoss()

	# Optimizer
	optimizer = optim.Adam(net.parameters(), lr = learning_rate)

	return (loss, optimizer)