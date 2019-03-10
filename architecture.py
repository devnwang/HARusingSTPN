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
import torch.optim.lr_scheduler
import Optical_Flow 
import STCB

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

inputs = Optical_Flow.opt_flow()

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
		for param in self.cnn1.parameters(), self.cnn2.parameters(), self.cnn3.parameters(), self.cnn4.parameters():
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
		self.stcb1 = STCB.CompactBilinearPooling(1024, 1024, 1024, 1024)
		self.stcb1.cuda()
		self.stcb1.train()
		self.out1 = stcb1(cnn2,cnn3,cnn4)

		self.stcb2 = STCB.CompactBilinearPooling(1024, 1024, 2048)
		self.stcb2.cuda()
		self.stcb2.train()
		self.out2 = stcb2(out1,cnn1)
		
		# Input channels = 7, output channels = 7
		self.conv1 = torch.nn.Conv2d(2048, 64, (7, 7), stride, padding)
		self.conv2 = torch.nn.Conv2d(64, 1, (7, 7), stride, padding)
		self.sm = torch.nn.Softmax2d()
		# self.pool3 = WeightedPool()
		# Substitute pooling layer
		self.pool3 = torch.nn.AvgPool2D((7, 7), stride, padding)

		# Intersection of streams
		self.stcb3 = STCB.CompactBilinearPooling(1024,1024,1024,4096)
		self.stcb3.cuda()
		self.stcb3.train()
		self.out3 = stcb3(pool1,pool2,pool3)
		
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

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
		

def createLossAndOptimizer(net, learning_rate = 0.001):
	# Loss function
	criterion = torch.nn.CrossEntropyLoss()

	# Optimizer
	optimizer = optim.Adam(net.parameters(), lr = learning_rate)

	scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

	return (criterion, optimizer, scheduler)

if __name__ == '__main__':
    # arrFrames = Optical_Flow.opt_flow()
    model_ft = train_model(STPN(), criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)
    # test = STPN(arrFrames)
    # print(test)