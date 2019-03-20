import os
import torch
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset

class Path():
	@staticmethod
	def db_dir(database):
		if database == 'ucf101':
			# Folder containing class labels
			root_dir = '/notebooks/storage/dataset/ucf'

			# Save preprocess data to output_dir
			output_dir = '/notebooks/storage/dataset/ucf_post_split'

			return root_dir, output_dir
		elif database == 'hmdb51':
			root_dir = '/notebooks/storage/dataset/hmdb'
			output_dir = '/notebooks/storage/dataset/hmdb_post_split'

			return root_dir, output_dir
		else:
			print('Database {} not available.'.format(database))
			raise NotImplementedError

	@staticmethod
	def model_dir():
		return '/path/to/Models/c3d-pretrained.pth'

class VideoDataset(Dataset):
	def __init__(self, dataset = 'ucf101', split = 'train', preprocess = False):
		self.root_dir, self.output_dir = Path.db_dir(dataset)