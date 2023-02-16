import torch 
from torch import nn

import 	tensorflow as tf
from 	tensorflow import keras
from 	tensorflow.keras import layers

import 	numpy as np
import 	matplotlib.pyplot as plt
from 	sklearn.model_selection import train_test_split

import 	sys


## ------------------------------------------------------------------------------
# Calculate the average standard deviation, more specifically the root 
# mean-sqaure calculation, to test goodness of fit
# -------------------------------------------------------------------------------
# Input:
	# K 			integer 		number of nuclei used in the prediction
	# delME_FRDM 	matrix			
	# delME_mod 	matrix 			 
# -------------------------------------------------------------------------------
# Output:
# -------------------------------------------------------------------------------
def avg_sd(K, delME_FRDM, delME_mod):

	# calculate the difference between model and FRDM for each nuclei
	# corresponding to an index 
	del_total = (delME_FRDM - delME_mod)**2

	# take the sum of all the differences squared
	sum_2 = del_total.sum()

	# divide the total by the total number of nuclei
	sigma_rms = np.sqrt(sum_2/K)

	print("")
	print("The RMS Standard Deviation is: " + str(sigma_rms) + " MeV")
# -------------------------------------------------------------------------------



# FEEDFORWARD NEURAL NETWORK APPROACHES 

## ------------------------------------------------------------------------------
class tf_NeuralNetwork:
## ------------------------------------------------------------------------------

	# --------------------------------------------------------------------------
	# Create a model that maps a vector of input features to a singular output
	# --------------------------------------------------------------------------
	# Input:
		# input_num 	integer 	number of units in the input layer
		# output_num 	integer  	number of units in the output layer
		# layer_num 	integer 	number of hidden layers in the network
		# hid_units 	array 		array corresponing to the # of units per layer
	# --------------------------------------------------------------------------
	# Output:
	# --------------------------------------------------------------------------
	def __init__(self, input_num, output_num, layer_num, hid_units):
		self.model = keras.Sequential()

		# Set the activation function for the hidden layers
		# -------------------------------------------------------------------
		act_func = "relu"
		# -------------------------------------------------------------------

		# Setting each layer as a Dense layer - each of the neurons of the dense 
		# layers receives input from all neurons of the previous layer
		# -------------------------------------------------------------------
		# Initial layer containing the a linear unit 
		self.model.add(keras.layers.InputLayer(input_shape=(input_num,)))

		# Hidden layers containing 60 units, with the ReLU activation function (or tanh)
		for l in range(layer_num):
			self.model.add(keras.layers.Dense(units = hid_units[l], \
						   activation = act_func, name='Hidden_Layer_'+str(l+1)))
	
		# Final output linear layer, since the prediction is continuous
		self.model.add(keras.layers.Dense(units = output_num, \
						   activation = 'linear', name='Output_Layer'))
		# --------------------------------------------------------------------

		# Compile the model. Uses the adam optimizer
		# -------------------------------------------------------------------
		# Compile. Loss function - mean square error
		self.mod_loss = 'mse'
		self.model.compile(loss=self.mod_loss, optimizer='adam')
		# -------------------------------------------------------------------
	## --------------------------------------------------------------------------

	
	# --------------------------------------------------------------------------
	# Display the model summary
	# --------------------------------------------------------------------------
	# --------------------------------------------------------------------------
	def display(self):
		self.model.summary()
	# --------------------------------------------------------------------------


	# --------------------------------------------------------------------------
	# Train the model
	# --------------------------------------------------------------------------
	# Input:
		# x_data 	array 		input training data 
		# y_data 	array 		corresponding outputs to the training data
		# runs 		integer		number of epochs (iterations)
		# p 		integer 	patience, controls when model "converges" 
		# 						(no improvement)
		# verbosity integer 	0 - no update, 1 - progress bar, 
		# 						2 - one line per epoch
	# --------------------------------------------------------------------------
	# Output:
	# --------------------------------------------------------------------------
	def fit(self, x_data, y_data, runs, p, verbosity):
		# split the input data into random train and test datasets
			# random_state - pass an int for reproducible output across multiple 
			# 				 function calls
		# ----------------------------------------------------
		x_train, x_val, y_train, y_val = \
		train_test_split(x_data, y_data, test_size = 0.2, random_state = 1)
		# ----------------------------------------------------

		# This callback will stop the training when there is no 
		# improvement in the val_loss after p consecutive epochs
		# ----------------------------------------------------
		callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=p)
		# ----------------------------------------------------

		# fit training data and evaluate testing data
		# ----------------------------------------------------
		res = self.model.fit(x_train, y_train, epochs = runs, \
			 				 callbacks=[callback], verbose = verbosity, \
			 				 validation_data = (x_val, y_val))
		# ----------------------------------------------------

		# save training and testing outputs for plotting
		# ----------------------------------------------------
		self.epochs = np.arange(1, len(res.history['val_loss'])+1, 1)
		self.loss, self.val_loss = res.history['loss'], res.history['val_loss'] 
		# ----------------------------------------------------
	# --------------------------------------------------------------------------

	
	# --------------------------------------------------------------------------
	# Compute the predictions for non-trained values
	# --------------------------------------------------------------------------
	# Input:
		# x_data	array 		input testing data 
	# --------------------------------------------------------------------------
	# Output:
		# y_data 	array 		predictions using the model  
	# --------------------------------------------------------------------------
	def predict(self, x_data):
		return self.model.predict(x_data)
	# --------------------------------------------------------------------------


	# --------------------------------------------------------------------------
	# Save the trained model to a path
	# --------------------------------------------------------------------------
	# Input:
		# filename 		string 		name of the model file (.mod)
		# output_path	string 		path of the output directory 
	# --------------------------------------------------------------------------
	# Output:
		# 		 		file 		containing the trained model
	# --------------------------------------------------------------------------
	def save(self, filename, output_path):
		self.model.save(output_path+filename)
	# --------------------------------------------------------------------------

	
	# --------------------------------------------------------------------------
	# Display the loss of training the model 
	# --------------------------------------------------------------------------
	# Input:
		# epochs		array 	number of iterations
		# loss 			array 	the values for the training loss per epoch
		# val_loss 		array 	the values for the testing loss per epoch
	# --------------------------------------------------------------------------
	# Output:
	# --------------------------------------------------------------------------
	def plot_loss(self):
		fig, ((ax1)) = plt.subplots(nrows=1, ncols=1)

		# plot training and testing loss
		# ------------------------------------
		ax1.plot(self.epochs, self.loss, 'r', c='b', linewidth=.4, \
				 label='training')
		ax1.plot(self.epochs, self.val_loss, 'r', c='k', linewidth=.4, \
				 label='validation')
		# ------------------------------------

		# Plot settings 
		# ------------------------------------
		# set labels
		ax1.set_xlabel('epochs', size = 12)
		ax1.set_ylabel(self.mod_loss + ' loss', size = 12)

		# remove spines
		ax1.spines['top'].set_visible(False)
		ax1.spines['right'].set_visible(False)

		# enable legend
		ax1.legend()

		# ------------------------------------

		plt.tight_layout()
		plt.show()
	# --------------------------------------------------------------------------
## ------------------------------------------------------------------------------





# Use a probabilistic neural network (single layer) (MDN)
# The idea is to assume the output data resemebles a mix of different probabiliy 
# distributions, os insted of calculating the output node, y, from input vec(x) we
# calculate the parameters of the probability distribution using our neural network. 
# - In this case we need mean (\mu) and variance (\sigma) for the Gaussian distribution.
# - these networks can capture probabilistic data much better than standard NN's


## ------------------------------------------------------------------------------
class pyt_NeuralNetwork:
## ------------------------------------------------------------------------------
	
	# --------------------------------------------------------------------------
	# Create a model that maps a vector of inputs to a set of outputs
	# --------------------------------------------------------------------------
	# Input:
		# net_type 		string 		refers to type of network nn or mdn
		# input_num 	integer 	number of features in the input layer
		# output_num 	integer  	number of features in the output layer
		# hid_units 	array 		array corresponing to the # of units per layer
	# --------------------------------------------------------------------------
	# Output:
	# --------------------------------------------------------------------------
	def __init__(self, net_type, input_num, output_num, hid_units):

		# Setting each layer - each of the neurons of the dense 
		# layers receives input from all neurons of the previous layer
		# ------------------------------------------------------------------
		# initialize the model
		# neural network with on hidden layer
		self.model = nn.Sequential(
					 # Initial layer containing the a linear unit 
					 nn.Linear(input_num, hid_units[0]),
					 # Hidden layer using the non-linear
					 	# ReLU activation function (or tanh)
		    		 nn.ReLU(),
		    		 # Final output linear layer, since the 
		    		 # prediction is continuous
		    		 nn.Linear(hid_units[0], output_num))
		# -------------------------------------------------------------------

		# Compile the model. Uses the adam optimizer
		# -------------------------------------------------------------------
		# Compile. Loss function - mean square error
		self.criterion = torch.nn.MSELoss()
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.01)
		# -------------------------------------------------------------------
	# --------------------------------------------------------------------------


	# --------------------------------------------------------------------------
	# Train the model
	# --------------------------------------------------------------------------
	# Input:
		# x_data 		numpy array 	input training data 
		# y_data 		numpy array 	corresponding outputs to the training data
		# num_epochs 	integer			number of epochs (iterations)
		# verbosity 	integer 		0 - no update, 1 - progress bar
	# --------------------------------------------------------------------------
	# Output:
	# --------------------------------------------------------------------------
	def fit(self, x_data, y_data, num_epochs, p, verbosity = 0):

		# split into training and validation set
		self.x_train, x_val, self.y_train, y_val = \
		train_test_split(x_data, y_data, test_size = 0.2, random_state = 1)

		# in order to get correct input dimensions
		x = torch.Tensor(self.x_train)
		y = torch.Tensor(self.y_train).unsqueeze(1)

		# training losses list
		self.epochs, self.losses = [], []

		print('')
		# Train the model
		for epoch in range(num_epochs):
			# Forward pass: Compute predicted y by passing x to the model
			self.y_pred = self.model(x)

			# compute and print loss
			loss = self.criterion(self.y_pred, y)

			if verbosity == 1:
				print('epoch: ', epoch,' loss: ', loss.item())

			# return validation and training loss
			# store training loss per epoch for the output plots
			self.epochs.append(epoch)
			self.losses.append(loss.item())

			# Zero gradients, perform a backward pass, and update the weights.
			self.optimizer.zero_grad()

			# perform a backward pass (backpropagation)
			loss.backward()

			# Update the parameters
			self.optimizer.step()
	# --------------------------------------------------------------------------


	# --------------------------------------------------------------------------
	# Display the model summary
	# --------------------------------------------------------------------------
	# --------------------------------------------------------------------------
	def display(self, input_size):
		from torchsummary import summary
		summary(self.model, (input_size))
	# --------------------------------------------------------------------------


	# --------------------------------------------------------------------------
	# Compute the predictions for non-trained values
	# --------------------------------------------------------------------------
	# Input:
		# x_data	numpy array 		input testing data 
	# --------------------------------------------------------------------------
	# Output:
		# y_data 	numpy array 		predictions using the model  
	# --------------------------------------------------------------------------
	def predict(self, x_data):
		# convert to pytorch tensor
		x = torch.Tensor(x_data)
		# predict data
		y = self.model(x)

		# convert back to numpy array
		y_pred = y.detach().cpu().numpy()

		return y_pred
	# --------------------------------------------------------------------------


	# --------------------------------------------------------------------------
	# Save the model to a path
	# --------------------------------------------------------------------------
	# Input:
		# filename 		string 		name of the model file (.mod)
		# output_path	string 		path of the output directory 
	# --------------------------------------------------------------------------
	# Output:
		# 		 		file 		containing the trained model
	# --------------------------------------------------------------------------
	def save(self, filename, output_path):
		torch.save(self.model.state_dict(), output_path+filename)
	# --------------------------------------------------------------------------


	# --------------------------------------------------------------------------
	# Load the model from a path
	# --------------------------------------------------------------------------
	# Input:
		# input_path	string 		input path of the file
	# --------------------------------------------------------------------------
	# Output:
		# 		 		model 		containing the trained model
	# --------------------------------------------------------------------------
	# def load(self, input_path):
	# 	return model.load_state_dict(torch.load(input_path))
	# --------------------------------------------------------------------------


	# --------------------------------------------------------------------------
	# Display the loss of training the model 
	# --------------------------------------------------------------------------
	# Input:
		# epochs		array 	number of iterations
		# loss 			array 	the values for the training loss per epoch
		# val_loss 		array 	the values for the testing loss per epoch
	# --------------------------------------------------------------------------
	# Output:
	# --------------------------------------------------------------------------
	def plot_loss(self):
		fig, ((ax1)) = plt.subplots(nrows=1, ncols=1)

		# plot training and testing loss
		# ------------------------------------
		ax1.plot(self.epochs, self.losses, 'r', c='b', linewidth=.4, \
				 label='training')
		# ax1.plot(self.epochs, self.val_loss, 'r', c='k', linewidth=.4, \
		# 		 label='validation')
		# ------------------------------------

		# Plot settings 
		# ------------------------------------
		# set labels
		ax1.set_xlabel('epochs', size = 12)
		ax1.set_ylabel('loss', size = 12)

		# remove spines
		ax1.spines['top'].set_visible(False)
		ax1.spines['right'].set_visible(False)

		# enable legend
		ax1.legend()
		# ------------------------------------

		plt.tight_layout()
		plt.show()
	# --------------------------------------------------------------------------

## ------------------------------------------------------------------------------



