import os
import torch
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split
import re
import optical_flow

class Path():
	@staticmethod
	def db_dir(database):
		if database == 'ucf101':
			# Folder containing class labels
			root_dir = '/notebooks/storage/dataset/ucf'

			# Local testing purposes
			# root_dir = r'C:\Users\Nikki Wang\Desktop\ucf'

			# Save preprocess data to output_dir
			output_dir = '/notebooks/storage/dataset/new_ucf_post_split'

			# Local testing purposes
			# output_dir = r'C:\Users\Nikki Wang\Desktop\ucf_post_split'

			return root_dir, output_dir
		elif database == 'hmdb51':
			root_dir = '/notebooks/storage/dataset/hmdb'
			output_dir = '/notebooks/storage/dataset/hmdb_post_split'

			return root_dir, output_dir
		else:
			print('Database {} not available.'.format(database))
			raise NotImplementedError

	# @staticmethod
	# def model_dir():
	# 	return '/path/to/Models/c3d-pretrained.pth'

class VideoDataset(Dataset):
# class VideoDataset():
	def __init__(self, dataset = 'ucf101', split = 'train', preprocess = False):
		self.root_dir, self.output_dir = Path.db_dir(dataset)
		folder = os.path.join(self.output_dir, split)
		self.split = split

		if not self.check_integrity():
			raise RuntimeError('Dataset not found or corrupted.' +
				' Please download it from the official website.')

		if (not self.check_preprocess()) or preprocess:
			print('Preprocessing of {} dataset. This will take long, but it will be done only once.'.format(dataset))
			self.preprocess()

		# Obtain all the filenames of files inside all the class folders
		# Goes through each class folder one at a time
		self.fnames, labels = [], []
		for label in sorted(os.listdir(folder)):
			for fname in os.listdir(os.path.join(folder, label)):
				self.fnames.append(os.path.join(folder, label, fname))
				labels.append(label)

		assert len(labels) == len(self.fnames)
		print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

		# Prepare a mapping between the label names (strings) and indices (ints)
		self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}

		# Convert the list of label names into an array of label indices
		self.label_array = np.array([self.label2index[label] for label in labels], dtype = int)

		path = '/notebooks/storage/dataset'
		if dataset == "ucf101":
			# Local testing purposes
			# path = r'C:\Users\Nikki Wang\Desktop'
			if not os.path.exists(os.path.join(path, 'ucf_labels.txt')):
				with open(os.path.join(path, "ucf_labels.txt"), 'w') as f:
					for id, label in enumerate(sorted(self.label2index)):
						f.writelines(str(id + 1) + ' ' + label + '\n')
		elif dataset == "hmdb51":
			if not os.path.exists(os.path.join(path, 'hmdb_labels.txt')):
				with open(os.path.join(path, 'hmdb_labels.txt'), 'w') as f:
					for id, label in enumerate(sorted(self.label2index)):
						f.writelines(str(id + 1) + ' ' + label + '\n')

	def __len__(self):
		return len(self.fnames)

	def check_integrity(self):
		if not os.path.exists(self.root_dir):
			return False
		else:
			return True

	def check_preprocess(self):
		if not os.path.exists(self.output_dir):
			return False
		elif not os.path.exists(os.path.join(self.output_dir, 'train')):
			return False

		return True

	def gen_train_test_split(self, dataset = 'ucf101', tlist = 1):
		# TODO: Read the train/test split text and split the data
		# self.root_dir = '/notebooks/storage/datasets/ucf'
		# self.output_dir = '/notebooks/storage/datasets/ucf_post_split'
		trainlist = 'trainlist0' + str(tlist) + '.txt'
		testlist = 'testlist0' + str(tlist) + '.txt'

		train_list, test_list = [], []

		# Read the train list
		direct = "/notebooks/storage/dataset"
		# Local testing purposes
		# direct = r'C:\Users\Nikki Wang\Desktop'
		
		if not os.path.exists(os.path.join(direct, trainlist)):
			print("Train list not available. Please include it in the dataset folder.")
		else:
			# files.readlines([sizehint])
			with open(os.path.join(os.path.join(direct, trainlist))) as f:
				train_list = f.readlines()

		# Read the test list
		if not os.path.exists(os.path.join(direct, testlist)):
			print("Test list not available. Please include it in the dataset folder.")
		else:
			with open(os.path.join(direct, testlist)) as f:
				test_list = f.readlines()

		# Grab only the video file names
		for files in range(len(train_list)):
			train_list[files] = train_list[files].split('avi')[0] + "avi"
		for files in range(len(test_list)):
			test_list[files] = test_list[files].split('avi')[0] + "avi"

		# Make trainlist and testlist frozen sets
		train_list = frozenset(train_list)
		test_list = frozenset(test_list)

		# Split train/test sets
		for folder in os.listdir(self.root_dir):
			# filepath = /notebooks/storage/datasets/ucf/class
			# LTP: root_dir = C:/Users/Nikki Wang/Desktop/ucf
			file_path = os.path.join(self.root_dir, folder)		# C:/Users/Nikki Wang/Desktop/ucf/class
			# video_files = list of videos in class folder
			video_files = [name for name in os.listdir(file_path)]

			# train/test_dir = /notebook/storage/datasets/ucf_post_split/train(test)/class
			# Local testing purposes: output_dir = C:/Users/Nikki Wang/Desktop/ucf_post_split
			train_dir = os.path.join(self.output_dir, 'train', folder)	# C:/Users/Nikki Wang/Desktop/ucf_post_split/train/action
			test_dir = os.path.join(self.output_dir, 'test', folder)	# C:/Users/Nikki Wang/Desktop/ucf_post_split/test/action

			# Creates folder for action in train/test split
			if not os.path.exists(train_dir):
				os.mkdir(train_dir)
			if not os.path.exists(test_dir):
				os.mkdir(test_dir)

			# train_list = ['ApplyEyeMakeup/v_ApplyEyeMakeup_g08_c01.avi', ...]
			for video in train_list:
				action = video.split('/')[0]
				if action == folder:
					self.process_video(video, train_dir)

			for video in test_list:
				action = video.split('/')[0]
				if action == folder:
					self.process_video(video, test_dir)


	def preprocess(self):
		if not os.path.exists(self.output_dir):
			os.mkdir(self.output_dir)
			os.mkdir(os.path.join(self.output_dir, 'train'))
			os.mkdir(os.path.join(self.output_dir, 'test'))

		# Split train/val/test sets
		# for file in os.listdir(self.root_dir):
		# 	file_path = os.path.join(self.root_dir, file)
		# 	video_files = [name for name in os.listdir(file_path)]

		# 	train_and_valid, test = train_test_split(video_files, test_size = 0.2, random_state = 42)
		# 	train, val = train_test_split(train_and_valid, test_size = 0.2, random_state = 42)

		# 	train_dir = os.path.join(self.output_dir, 'train', file)
		# 	val_dir = os.path.join(self.output_dir, 'val', file)
		# 	test_dir = os.path.join(self.output_dir, 'test', file)

		# 	if not os.path.exists(train_dir):
		# 		os.mkdir(train_dir)
		# 	if not os.path.exists(val_dir):
		# 		os.mkdir(val_dir)
		# 	if not os.path.exists(test_dir):
		# 		os.mkdir(test_dir)

		# 	for video in train:
		# 		self.process_video(video, file, train_dir)

		# 	for video in val:
		# 		self.process_video(video, file, val_dir)

		# 	for video in test:
		# 		self.process_video(video, file, test_dir)

		self.gen_train_test_split(dataset = 'ucf101', tlist = 1)

		print('Preprocessing finished.')

	def process_video(self, video, save_dir):
		# TODO: Modify for our archirtecture
		# Initialize a VideoCapture object to read video data into a numpy array
		# video_filename = video.split('.')[0]
		# if not os.path.exists(os.path.join(save_dir, video_filename)):
		# 	os.mkdir(os.path.join(save_dir, video_filename))

		# cap = cv.VideoCapture(os.path.join(self.root_dir, action_name, video))

		# frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
		# frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
		# frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

		# TODO: Extract frames and save them

		# video = 'action_name/video'
		# save_dir = train_dir = C:/Users/Nikki Wang/Desktop/ucf_post_split/train/action
		# action_name = video.split('/')[0]
		video_filename = video.split('/')[1].split('.')[0]
		# video_filename = video.split('.')[0]
		
		# If the folder for the video doesn't exist, make it
		if not os.path.exists(os.path.join(save_dir, video_filename)):
			os.mkdir(os.path.join(save_dir, video_filename))

		# cap = cv.VideoCapture(os.path.join(self.root_dir, action_name, video))
		# C:/Users/Nikki Wang/Desktop/ucf/action_name/video
		# print("cap = {}".format(str(os.path.join(self.root_dir, video))))
		cap = cv.VideoCapture(os.path.join(self.root_dir, video))
		save_path = os.path.join(save_dir, video_filename)
		optical_flow.getInputs(cap, save_path)

		cap.release()

	def randomflip(self, buffer):
		'''
		Horizontally flip the give image and ground truth randomly with a probability of 0.5
		'''

		if np.random.random() < 0.5:
			for i, frame in enumerate(buffer):
				frame = cv.flip(buffer[i], flipCode = 1)
				buffer[i] = cv.flip(frame, flipCode = 1)

		return buffer

	def normalize(self, buffer):
		for i, frame in enumerate(buffer):
			frame -= np.array([[[90.0, 98.0, 102.0]]])
			buffer[i] = frame

		return buffer

if __name__ == "__main__":
	train_data = VideoDataset(dataset = 'ucf101', split = 'train', preprocess = True)
