import os
import sys
import csv
import numpy as np
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

from sourcerer import SourceRegLoss
from cnn_model import ConvModel
from sits_data import SITSData

def main():

	source_tile = sys.argv[1]
	target_tile = sys.argv[2]
	run_no = sys.argv[3]
	polygons = sys.argv[4]

	main_path = "/home/bluc0001/nc23_scratch/ben/"
	exp_name = "Sourcerer_s_"+source_tile+"_t_"+target_tile
	NO_CLASSES = 30
	data_path = "/home/bluc0001/nc23_scratch/ben/data/"
	NO_BANDS = 10
	results_path = main_path+"Results/"

	seed = 18
	np.random.seed(seed)
	torch.manual_seed(seed)

	##### INITIALISE MODEL #####
	cnn = ConvModel()

	##### LOSS FUNCTION AND OPTIMIZER #####
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(cnn.parameters())

	model_filepath = results_path+exp_name+"_run_"+run_no+"_model.pth"

	if os.path.isfile(model_filepath):
		print("Model Found, skipping training on Source")
		cnn = torch.load(model_filepath)
	else:
		print("Training on Source Train data...")
		##### LOADING SOURCE TRAINING DATA #####
		source_train_data_file = data_path+source_tile+"/"+source_tile+"_train.csv"
		source_train = SITSData(source_train_data_file, n_channels=NO_BANDS)
		source_train_generator = data.DataLoader(source_train, batch_size=32, drop_last=True)

		##### TRAIN MODEL #####
		cnn.train()

		correct_val = 0
		total_val = 0
		loss_list = []

		for i, (X, y) in enumerate(source_train_generator):

			# Run the forward pass
			predictions = cnn(X.float())
			loss = criterion(predictions, y)
			loss_list.append(loss.item())

			# Backprop and optimise
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			# Track the accuracy
			total_val += y.size(0)
			_, predicted = torch.max(predictions.data, dim=1)
			correct_val += (predicted == y).sum().item()

		train_accuracy = correct_val/total_val
		print("Train Accuracy: ", train_accuracy)

		del source_train, source_train_generator
		torch.save(cnn, model_filepath)

	print("Training on "+str(polygons)+" Target Polygons")
	if int(polygons)==0:
		target_train_qty = 0
		epochs_required = 0
		reg_constant = 0
	else:
		tfr_model_filename = results_path+exp_name+"_run_"+run_no+"_pgns_"+str(polygons)+"_model.pth"

		if os.path.isfile(tfr_model_filename):
			print("Model found... exiting...")
			sys.exit()

		for module in cnn.modules():
			if isinstance(module, nn.BatchNorm1d):
				module.eval()

		optimizer = optim.Adam(cnn.parameters())

		#### LOADING TARGET TRAIN DATA  #####
		target_train_data_file = data_path+target_tile+"/"+target_tile+"_subsets_run_poly/"+"run_"+run_no+"_polygons_"+str(polygons)+".csv"
		target_train_qty = sum(1 for line in open(target_train_data_file))-1
		print("Target train qty: ", target_train_qty)
		target_train = SITSData(target_train_data_file, n_channels=NO_BANDS)
		batch_sz = 32
		print("Batch Size: ", batch_sz)
		target_train_generator = data.DataLoader(target_train, batch_size = batch_sz)

		source_reg_loss = SourceRegLoss(cnn, target_train_qty)

		no_updates = 5000
		print("No updates: ", no_updates)
		epochs_required = math.ceil(no_updates * batch_sz / target_train_qty)
		print("No epochs: ", epochs_required)

		for epoch in range(epochs_required):
			for i, (X, y) in enumerate(target_train_generator, 0):

				# Run the forward pass
				predictions = cnn(X.float())
				loss = source_reg_loss(predictions, y, cnn)
				print("Loss: ", loss.data)

				# Backprop and optimise
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

		torch.save(cnn, tfr_model_filename)
		del target_train, target_train_generator

	#### LOADING TARGET TEST DATA  #####
	target_test_data_file = data_path+target_tile+"/"+target_tile+"_test.csv"
	target_test = SITSData(target_test_data_file, n_channels=NO_BANDS)
	target_test_generator = data.DataLoader(target_test, batch_size=10_000)

	##### PREDICT TEST DATA #####
	cnn.eval()
	with torch.no_grad():
		correct_test = 0
		total_test = 0
		for i, (X, y) in enumerate(target_test_generator):
			predictions = cnn(X.float())
			_, predicted = torch.max(predictions.data, dim=1)
			total_test += y.size(0)
			correct_test += (predicted == y).sum().item()

		test_accuracy = correct_test/total_test

	print("--------------------------------------------------------")
	print("Total test acc: ", test_accuracy)
	print("--------------------------------------------------------")


if __name__ == "__main__":
	main()
