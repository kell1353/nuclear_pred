# import torch 
import 	tensorflow as tf
from 	tensorflow import keras
from 	tensorflow.keras import layers

import 	numpy as np
import 	matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import 	sys


## ------------------------------------------------------------------------------
# Calculate the average standard deviation, more specifically the root 
# mean-sqaure calculation, to test goodness of fit
# -------------------------------------------------------------------------------
# Input:
	# K 			integer 		number of nuclei used in the prediction
	# delME_AME 	array 			
	# delME_MDN 	array 			 
# -------------------------------------------------------------------------------
# Output:
# -------------------------------------------------------------------------------
def avg_sd(K, delME_AME, delME_MDN):

	sum = 0
	for i in range(K):
		sum += (delME_AME[i] - delME_MDN[i])**2

	return np.sqrt(sum/K)
# -------------------------------------------------------------------------------



# FEEDFORWARD NEURAL NETWORK APPROACHES 

## ------------------------------------------------------------------------------
class NeuralNetwork:
## ------------------------------------------------------------------------------

	# --------------------------------------------------------------------------
	# Create a model that maps a vectot of inputs to a singular output
	# --------------------------------------------------------------------------
	# Input:
		# mod_loss 		string 		name of the loss used in training the model
		# input_num 	integer 	number of units in the input layer
		# output_num 	integer  	number of units in the output layer
		# layer_num 	integer 	number of hidden layers in the network
		# hid_units 	array 		array corresponing to the # of units per layer
	# --------------------------------------------------------------------------
	# Output:
	# --------------------------------------------------------------------------
	def __init__(self, mod_loss, input_num, output_num, layer_num, hid_units):
		self.model = keras.Sequential()

		# Setting each layer as a Dense layer - each of the neurons of the dense 
		# layers receives input from all neurons of the previous layer
		# -------------------------------------------------------------------
		# Initial layer containing the a linear unit 
		# self.model.add(keras.layers.Dense(units = input_num, activation = 'linear', \
		# 								  input_shape=(input_num,), name='Input_Layer'))
		self.model.add(keras.layers.InputLayer(input_shape=(input_num,), name='Input_Layer'))

		# Hidden layers containing 60 units, with the ReLU activation function (or tanh)
		for l in range(layer_num):
			self.model.add(keras.layers.Dense(units = hid_units[l], activation = 'relu', name='Hidden_Layer_'+str(l+1)))
	
		# Final output linear layer, since the prediction is continuous
		self.model.add(keras.layers.Dense(units = output_num, activation = 'linear', name='Output_Layer'))
		# --------------------------------------------------------------------

		# Compile. Use the loss function - mean square error, and optimization - adam
		self.mod_loss = mod_loss
		self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
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
		# verbosity integer 	0 - no update, 1 - progress bar, 2 - one line per epoch
	# --------------------------------------------------------------------------
	# Output:
	# --------------------------------------------------------------------------
	def fit(self, x_data, y_data, runs, verbosity):
		# split the input data into random train and test datasets
			# random_state - pass an int for reproducible output across multiple 
			# 				 function calls
		# ----------------------------------------------------
		x_train, x_val, y_train, y_val = \
		train_test_split(x_data, y_data, test_size = 0.5, random_state = 1)
		# ----------------------------------------------------

		# This callback will stop the training when there is no 
		# improvement in the val_loss for twenty consecutive epochs
		# ----------------------------------------------------
		callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)
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

		# note: in regresssion analysis such as mse loss 
		# accuracy is meaningless since mse loss can be used 
		# as the performance evaluater
		self.accuracy, self.val_accuracy = res.history['accuracy'], res.history['val_accuracy']
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
		ax1.plot(self.epochs, self.loss, 'r', c='b', linewidth=.4, label='training')
		ax1.plot(self.epochs, self.val_loss, 'r', c='k', linewidth=.4, label='validation')
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


	# --------------------------------------------------------------------------
	# Display the accuracy of training the model (if possible)
	# --------------------------------------------------------------------------
	# Input:
		# epochs		array 	number of iterations
		# accuracy 		array 	the values for the training acccuracy per epoch
		# val_accuracy 	array 	the values for the testing acccuracy per epoch
	# --------------------------------------------------------------------------
	# Output:
	# --------------------------------------------------------------------------
	# --------------------------------------------------------------------------
	def plot_accuracy(self):
		fig, ((ax1)) = plt.subplots(nrows=1, ncols=1)

		# plot training and testing accuracy
		# ------------------------------------
		ax1.plot(self.epochs, self.accuracy, 'r', c='b', linewidth=.4, label='training')
		ax1.plot(self.epochs, self.val_accuracy, 'r', c='k', linewidth=.4, label='validation')
		# ------------------------------------

		# Plot settings 
		# ------------------------------------
		# set labels
		ax1.set_xlabel('epochs', size = 12)
		ax1.set_ylabel('accuracy', size = 12)

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


# Our MDN in PyTorch uses an Adam optimizer with
# a learning rate of 0.001 (the default), with a hyperbolic
# tangent as the activation function. The weights of the
# NN are initialized randomly. The MDN was found to
# be converged after training for 100,000 epochs, which is
# used throughout this work.

# class MDN:




## ------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
def example_prob():
	# https://lucidar.me/en/neural-networks/curve-fitting-nonlinear-regression/
	# f(x) = .1*x*cos(x) + .1*constant
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
	nn = NeuralNetwork(1, 1, 2, [60, 60])
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
# -------------------------------------------------------------------------------
