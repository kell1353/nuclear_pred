import numpy as np
import matplotlib.pyplot as plt
from   sklearn.model_selection import train_test_split

import setup as su
import ml_lib as ml
import plot as pl

import mdn
import torch

import sys


## -------------------------------------------------------------------------------
# Trains a model to calculate mass excess using both a neural network and a mixture 
# density network for different levels of input information. Saves all of the 
# models and plots concerned with only the nuclei measured in the AME (2016) to 
# the specified output directory
# --------------------------------------------------------------------------------
# Input:
	# output_path 		string 		path of the output directory 
	# full_data 		arrays		data from the AME + features data set
	# FRDM_full_data	array 		data from the FRDM model (containing observables)
# --------------------------------------------------------------------------------
# Output: 				plots and model information
# --------------------------------------------------------------------------------
def nn_mdn_comparison(output_path, full_data, FRDM_full_data):

	networks = ['nn', 'mdn']
	model_nums = [2, 6, 8, 10, 12]
 
	for network in networks:
		for m_num in model_nums:

			# get model data
			M = su.model_select("AME", m_num, full_data) 
			title = "M"+str(m_num)+"_"+network

			# Convert dataframe values to numpy arrays
			x_data, y_data = M[0].to_numpy(), M[1].to_numpy()


			# Setup the network 
			# -----------------------------------------------------
			# num. of units in input, output and hidden layers
			input_l, output_l, hid_l = m_num, 1, [10] 

			# Creates the neural network class
			if network == "mdn":
				num_gaussians = 4
				nn = ml.pyt_MixtureDensityNetwork(input_l, output_l, hid_l, num_gaussians)
			else:
				nn = ml.pyt_NeuralNetwork(network, input_l, output_l, hid_l)

			# Display the model summary 
			nn.display((0, m_num))

			# Train the model
			nn.fit(x_data, y_data, 20000, 500, 0)

			# Save the model to .txt file
			# nn.save(title+"_model", output_path)
			# -----------------------------------------------------


			# using the model to calculate the mass excess for all the available 
			# N, Z in the AME compared to the FDRM (testing) dataset (Fig. 1)
			# --------------------------------------------------------------------
			y_FRDM = FRDM_full_data['ME_F'].to_numpy()

			# Use the model on the nuclei NOT included in the AME_data
			# and compare to the FDRM values
			# -----------------------------------------------------
			y_mod = nn.predict(x_data) 
			# -----------------------------------------------------

			# setup the matrices for the datasets 
			# -----------------------------------------------------
			frdm_M = pl.create_matrix(FRDM_full_data['Z'], FRDM_full_data['N'], y_FRDM)
			model_M = pl.create_matrix(full_data['Z'], full_data['N'], y_mod)
			# -----------------------------------------------------


			# plot the output heatmap comparison
			# -----------------------------------------------------
			del_M = pl.matrix_sub(model_M, frdm_M)

			# plot title
			p_filename = "model_"+title+".pdf"
			# plot
			pl.plot_output(title, p_filename, output_path, del_M)
			# -----------------------------------------------------
			# --------------------------------------------------------------------


			# MSE between model and FRDM for all the nuclei including those outside 
			# of experimental observations
			# --------------------------------------------------------------------
			# get model testing data
			M = su.model_select("FRDM", m_num, FRDM_full_data) 

			# Convert dataframe values to numpy arrays
			x_data_FRDM, y_data_FRDM = M[0].to_numpy(), M[1].to_numpy()
			K = len(x_data_FRDM) # Number of nuclei 

			# Use the model to predict mass excess for the nuclei including 
			# with values that are currently experimentally unknown
			# ----------------------------------------------------------------
			y_mod_FRDM = nn.predict(x_data_FRDM)
			model_M2 = pl.create_matrix(FRDM_full_data['Z'], FRDM_full_data['N'], y_mod_FRDM)

			# Calculate the rms difference between the model predictions and 
			# the FRDM predictions
			# ----------------------------------------------------
			ml.avg_sd(K, frdm_M, model_M2)
			# ----------------------------------------------------
			print('')
			# ----------------------------------------------------------------

		# TESTING: working some kinks out with the mdn
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

	# Plot the training results
	y_preds = [y.item() for y in y_pred ]

	plt.scatter(x_train, y_train, s = 4)
	plt.plot(x_train, y_preds, 'r', linewidth = .4)
	plt.show()
## -------------------------------------------------------------------------------



# https://notebook.community/hardmaru/pytorch_notebooks/mixture_density_networks
def mdn_example():
	from torch.autograd import Variable

	# generate sample data
	# ---------------------------------------------------------
	n_samples = 1000

	epsilon = np.random.normal(size=(n_samples))
	x_data = np.random.uniform(-10.5, 10.5, n_samples)
	y_data = 7*np.sin(0.75*x_data) + 0.5*x_data + epsilon

	mdnx_data, mdny_data = y_data, x_data

	x_test_data = np.linspace(-15, 15, n_samples)
	# ---------------------------------------------------------

	# Initialize the network
	# ---------------------------------------------------------
	input_num, output_num = 1, 1
	hid_units, num_gaussians = [20], 5
	mdn_model = ml.pyt_MixtureDensityNetwork(input_num, output_num, hid_units, \
	 								   		 num_gaussians)
	# ---------------------------------------------------------

	mdn_model.display((0, 1))

	# Train the network
	# ---------------------------------------------------------
	num_epochs = 10000
	p = 200
	mdn_model.fit(mdnx_data, mdny_data, num_epochs, p, verbosity = 0)
	# Plot loss
	pl.plot_loss(mdn_model.epochs, mdn_model.losses, mdn_model.val_losses, "NLL")
	# ---------------------------------------------------------

	# Plot the training results
	# ---------------------------------------------------------
	pi_pred, sigma_pred, mu_pred = mdn_model.predict(x_test_data)

	# plot the data
	fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2)

	# FIRST plot
	ax1.scatter(mdnx_data, mdny_data, s=6, alpha = .2, c='black')
	for i in range(num_gaussians):
		# plot means and variances of the Gaussians
		ax1.plot(x_test_data, mu_pred[:,i], linewidth=.7, label="Mixture "+str(i))
		ax1.fill_between(x_test_data, mu_pred[:,i]-sigma_pred[:,i], \
						 mu_pred[:,i]+sigma_pred[:,i], alpha=0.1)

	# SECOND plot
	k = mdn.gumbel_sample(pi_pred)
	indices = (np.arange(n_samples), k)
	rn = np.random.randn(n_samples)
	samples = rn * sigma_pred[indices] + mu_pred[indices]

	# sample one Gaussian
	# samples = mdn.sample(torch.Tensor(pi_pred), torch.Tensor(sigma_pred), \
	# 					 torch.Tensor(mu_pred))

	ax2.scatter(mdnx_data, mdny_data, s=6, alpha = .2)
	ax2.scatter(x_test_data, samples, s=6, alpha = .2, color = 'red')
	
	ax1.legend()
	plt.show()
	# ---------------------------------------------------------














