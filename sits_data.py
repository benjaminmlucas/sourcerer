import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class SITSData(Dataset):

	def __init__(self, filepath, n_channels, nrows=-1):
		if nrows==-1:
			all_data = pd.read_csv(filepath, sep=',', header=0)
		else:
			all_data = pd.read_csv(filepath, sep=',', header=0, nrows=nrows)
		all_data = all_data.sample(frac=1).reset_index(drop=True)

		y_data = all_data.iloc[:,11]
		y_array = np.asarray(y_data.values, dtype='uint8')
		self.y = np.asarray([k-1 for k in y_array])

		# get class distribution
		unique, counts = np.unique(y_data, return_counts=True)
		self.class_counts = np.asarray((unique, counts)).T

		# reading min and max for normalisation
		min, max = self._readminmax()

		# preprocessing X data
		X_data = all_data.iloc[:,15:]
		X = X_data.values
		X = np.asarray(X, dtype='float32')
		X_rshp = X.reshape(X.shape[0],int(X.shape[1]/n_channels),n_channels)
		X_norm = self._normalise(X_rshp, min, max)
		self.X = np.swapaxes(X_norm, 1, 2)

	def __len__(self):
		return self.X.shape[0]

	def __getitem__(self, index):
		series = torch.from_numpy(self.X[index])
		return series, self.y[index]

	def _readminmax(self):
		file = "/home/bluc0001/nc23_scratch/ben/data/T31TEL/T31TEL_percentiles.csv"
		if not os.path.isfile(file):
			file = "/media/benny/Extra/data/T31TEL/T31TEL_percentiles.csv"
		minmax = np.loadtxt(file, delimiter = ",")
		min_pt = minmax[0,:]
		max_pt = minmax[1,:]
		return min_pt, max_pt

	def _normalise(self, X, min_pctl, max_pctl):
		return (X-min_pctl)/(max_pctl-min_pctl)

	def get_class_counts(self):
		return print(self.class_counts)
