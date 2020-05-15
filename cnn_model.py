import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class ConvModel(nn.Module):

	# input shape for Sentinel-2 is (n x 10 spectral bands x 37 timestamps)

	def __init__(self, no_channels=10):
		super().__init__()

		self.conv1 = nn.Sequential(
			nn.Conv1d(in_channels=no_channels, out_channels=64, kernel_size=5, padding=2),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Dropout(p=0.5)
		)

		self.conv2 = nn.Sequential(
			nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Dropout(p=0.5)
		)

		self.conv3 = nn.Sequential(
			nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
			nn.BatchNorm1d(64),
			nn.ReLU(),
			nn.Dropout(p=0.5)
		)

		# flatten output here n x 2368 (= 64 x 37)

		self.fc1 = nn.Sequential(
			nn.Linear(in_features=2368, out_features=256),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			nn.Dropout(p=0.5)
		)

		self.fc2 = nn.Linear(in_features=256, out_features=30)


	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = torch.flatten(x, start_dim=1)
		x = self.fc1(x)
		x = self.fc2(x)
		return x


if __name__ == "__main__":
	test_model = ConvModel()
	summary(test_model, input_size=(10, 37))
