import sys
import numpy as np
import matplotlib.pyplot as plt
from   sklearn.model_selection import train_test_split

import setup as su
import ML_lib as ml
import plot as pl


## -------------------------------------------------------------------------------
# Trains a model to calculate mass excess using both a neural network and a mixture 
# density network for different levels of input information. Saves all of the 
# models and plots concerned with only the nuclei measured in the AME (2016) to 
# the specified output directory
# --------------------------------------------------------------------------------
# Input:
	# output_path 		string 		path of the output directory 
	# full_data 		arrays		data from the AME + features data set
	# FRDM_data			array 		data from the FRDM model
# --------------------------------------------------------------------------------
# Output: 				plots and model information
# --------------------------------------------------------------------------------
def nn_mdn_comparison(output_path, full_data, FRDM_data):

	networks = ['nn', 'mdn']
	model_nums = [2, 6, 8, 10, 12]
 
	for network in networks:
		for m_num in model_nums:

			# get model data
			M = su.model_select(m_num, full_data) 
			title = "M"+str(m_num)+"_"+network

			# Convert dataframe values to numpy arrays
			x_data, y_data = M[0].to_numpy(), M[1].to_numpy()


			# Setup the network 
			# -----------------------------------------------------
			# number of hidden layers
			num_hl = 1
			# num. of units in input, output and hidden layers
			input_l, output_l, hid_l = m_num, 1, [10] 

			# Creates the neural network class
			nn = ml.tf_NeuralNetwork(network, input_l, output_l, num_hl, hid_l)
			# Display the model summary 
			nn.display()
			# Train the model
			nn.fit(x_data, y_data, 10000, 100, 0)
			# Save the model to .txt file
			nn.save(title+"_model", output_path)
			# -----------------------------------------------------
			# --------------------------------------------------------------------

			# using the model to calculate the mass excess for all the available 
			# N, Z in the AME compared to the FDRM (testing) dataset (Fig. 1)
			# --------------------------------------------------------------------
			y_FRDM = FRDM_data['ME_F'].to_numpy()

			# Use the model on the nuclei NOT included in the AME_data
			# and compare to the FDRM values
			# -----------------------------------------------------
			y_mod = nn.predict(x_data) 
			# -----------------------------------------------------

			# setup the matrices for the datasets 
			# -----------------------------------------------------
			frdm_M = pl.create_matrix(FRDM_data['Z'], FRDM_data['N'], y_FRDM)
			model_M = pl.create_matrix(full_data['Z'], full_data['N'], y_mod)
			# -----------------------------------------------------

			# plot the output heatmap comparison
			# -----------------------------------------------------
			del_M = pl.matrix_sub(model_M, frdm_M)

			# plot settings
			filename = "model_" + title + ".png"

			pl.plot_output(title, filename, output_path, del_M)
			# -----------------------------------------------------
			# --------------------------------------------------------------------

		sys.exit()
## -------------------------------------------------------------------------------



## -------------------------------------------------------------------------------
# Example problem fitting data for a non-linear sine function using TensorFlow
# FROM: https://lucidar.me/en/neural-networks/curve-fitting-nonlinear-regression/
	  # f(x) = .1*x*cos(x) + .1*constant
# --------------------------------------------------------------------------------
def TensorFlow_ex():
	
	# Setup function data f(x) = .1*x*cos(x) + .1*constant
	d_pts = 1000
	x_data = np.linspace(-10, 10, num=d_pts)
	y_data = 0.1*x_data*np.cos(x_data) + 0.1*np.random.normal(size=d_pts)

	rand = np.random.randint(0, d_pts, size=500)

	x_train, y_train = [], []
	for i in rand: 
		x_train.append(x_data[i])
		y_train.append(y_data[i]) 

	print('')
	# Creates the neural network class
	nn = tf_NeuralNetwork(1, 1, 2, [60, 60])
	# Display the model summary 
	nn.display()
	# Train the model
	nn.fit(x_train, y_train, 100)
	# Test the model
	y_pred = nn.predict(x_data)

	# Plot the testing results
	plt.scatter(x_data[::1], y_data[::1], s = 4)
	plt.plot(x_data, y_predicted, 'r', linewidth = .4)
	plt.show()
# --------------------------------------------------------------------------------



## -------------------------------------------------------------------------------
# same problem as with the TensorFlow example. only in function we are calling 
# PyTorch modules
## -------------------------------------------------------------------------------
def pytorch_ex():
	import torch
	from torch import nn
	
	# Setup function data f(x) = .1*x*cos(x) + .1*constant
	d_pts = 1000
	x_data = np.linspace(-10, 10, d_pts)
	y_data = 0.1*x_data*np.cos(x_data) + 0.1*np.random.normal(size=d_pts)

	# split into training and validation set
	x_train, x_val, y_train, y_val = \
		train_test_split(x_data, y_data, test_size = 0.2, random_state = 1)

	# sort x_train and y_train since plotting is dependent on order
	# zip
	z = zip(x_train, y_train)
	# sort the pairs
	sorted_pairs = sorted(z)
	# get sorted tuples back
	tuples = zip(*sorted_pairs)
	x_train, y_train = [ list(tuple) for tuple in  tuples]

	# in order to get correct input dimensions
	x = torch.FloatTensor(x_train).unsqueeze(-1) 
	y = torch.FloatTensor(y_train).unsqueeze(-1) 

	# initialize the model
	model = nn.Sequential(
			nn.Linear(1, 60),
	    	nn.ReLU(),
	    	nn.Linear(60, 60),
	    	nn.ReLU(),
	    	nn.Linear(60, 1))

	# Construct the loss function
	criterion = torch.nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=.01)

	print('')
	# Train the model
	for epoch in range(500):
	   # Forward pass: Compute predicted y by passing x to the model
	   y_pred = model(x)

	   # Compute and print loss
	   loss = criterion(y_pred, y)
	   print('epoch: ', epoch,' loss: ', loss.item())

	   # Zero gradients, perform a backward pass, and update the weights.
	   optimizer.zero_grad()

	   # perform a backward pass (backpropagation)
	   loss.backward()

	   # Update the parameters
	   optimizer.step()


	# use the model to predict valeus


	# Plot the training results
	y_preds = [y.item() for y in y_pred ]

	plt.scatter(x_train, y_train, s = 4)
	plt.plot(x_train, y_preds, 'r', linewidth = .4)
	plt.show()
## -------------------------------------------------------------------------------


# def mdn_net():
# 	import torch.nn as nn
# 	import torch.optim as optim
# 	import mdn

# 	# initialize the model
# 	model = nn.Sequential(
# 	    	nn.Linear(5, 6),
# 	    	nn.Tanh(),
# 	    	mdn.MDN(10, 7, 1)
# 	)
# 	optimizer = optim.Adam(model.parameters())

# 	# train the model
# 	for minibatch, labels in train_set:
# 	    model.zero_grad()
# 	    pi, sigma, mu = model(minibatch)
# 	    loss = mdn.mdn_loss(pi, sigma, mu, labels)
# 	    loss.backward()
# 	    optimizer.step()

# 	# sample new points from the trained model
# 	minibatch = next(test_set)
# 	pi, sigma, mu = model(minibatch)
# 	samples = mdn.sample(pi, sigma, mu)












