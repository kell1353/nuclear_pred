# import torch 
import 	tensorflow as tf
from 	tensorflow import keras
from 	tensorflow.keras import layers

import 	numpy as np
import 	matplotlib.pyplot as plt

import 	sys


# POSSIBLY EXPLORE MULTIPLE TYPES OF NEURAL NETWORKS--------------

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


## ------------------------------------------------------------------------------
# 
# -------------------------------------------------------------------------------
# Input:
	# 		 
# -------------------------------------------------------------------------------
# Output:
# -------------------------------------------------------------------------------
def minmax_norm(array):

	n = len(array)

	min_val = min(array) # minimum value of the array
	max_val = max(array) # maximum value of the array

	# the numerical range
	array_range = max_val - min_val

	# normalize the array to values betweeen 0 and 1
	norm = np.zeros(n)
	for i in range(n):
		norm[i] = (array[i] - min_val)/array_range

	return norm
# -------------------------------------------------------------------------------



# STANDARD MODELING APPROACH (LINEAR LEAST SQUARES)
	
## ------------------------------------------------------------------------------
# Linear least squares appoach to constructing a linear model
# -------------------------------------------------------------------------------
# Input:
	# num_rows 		integer 	number of different nuclei in the data
	# num_params 	integer 	number of parameters used to model the data
	# ME 			array 		the experimental mass excess for the nuclei
	# del_ME 		array 		the experimental uncertainty of the mass excess
	# lin_terms		matrix 		the features used in the model
# -------------------------------------------------------------------------------
# Output:
	# a 			array 		returns the optimal parameters for the model
# -------------------------------------------------------------------------------
def LLSQ(num_rows, num_params, ME, del_ME, lin_terms): 

	# Model: h(x) = sum_i=0^num_params a_i*f_i(N,Z) 
		# f - basis functions (defined in features) | a - weights

	# minimize the chi_squares function by setting the derivatives equal to zero.
	# doing this we end up needing to solve a system of equations. 
		# alpha * a = beta - > a = alpha^-1 * beta | a - set of parameters

	# Setup the arrays
	# --------------------------------------------------
	alpha = np.zeros((num_params, num_params))
	beta = np.zeros(num_params)
	# --------------------------------------------------

	# Construct the alpha and beta array used in the min. chis_squared 
	# ---------------------------------------------------
	for k in range(num_rows):

		# to avoid a divide by zero error at C12, 
		# since it works as the exp. baseline
		# -----------------------------------------
		if del_ME[k] == 0: continue
		# -----------------------------------------

		for i in range(num_params):
			for j in range(num_params):

				alpha[i,j] += ((lin_terms.iloc[k,i]*lin_terms.iloc[k,j])/(del_ME[k]**2.0))

			beta[i] += ((lin_terms.iloc[k,i]*ME[k])/(del_ME[k]**2.0))
	# ---------------------------------------------------

	# Invert the alpha matrix
	# ---------------------------------------------------
	inv_alpha = np.linalg.inv(alpha)

	# matrix multiply to get the model parameters
	a = inv_alpha @ beta 
	# ---------------------------------------------------

	return a 
# -------------------------------------------------------------------------------




# FEEDFORWARD NEURAL NETWORK APPROACHES 

## ------------------------------------------------------------------------------
class NeuralNetwork:
## ------------------------------------------------------------------------------

	# --------------------------------------------------------------------------
	# Create a model that maps a vectot of inputs to a singular output
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

		# Setting each layer as a Dense layer - each of the neurons of the dense 
		# layers receives input from all neurons of the previous layer
		# -------------------------------------------------------------------
		# Initial layer containing the a linear unit 
		self.model.add(keras.layers.Dense(units = input_num, activation = 'linear', input_shape=(input_num,), name='Input_Layer'))
		
		# Hidden layers containing 60 units, with the ReLU activation function
		for l in range(layer_num):
			self.model.add(keras.layers.Dense(units = hid_units[l], activation = 'relu', name='Hidden_Layer_'+str(l)))
	
		# Final output linear layer, since the prediction is continuous
		self.model.add(keras.layers.Dense(units = output_num, activation = 'linear', name='Output_Layer'))
		# --------------------------------------------------------------------

		# Compile. Use the loss function - mean square error, and optimization - adam
		self.model.compile(loss='mse', optimizer="adam")
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
		# returns	integer		number of epochs (iterations)
	# --------------------------------------------------------------------------
	# Output:
	# --------------------------------------------------------------------------
	def fit(self, x_data, y_data, runs):
		self.model.fit(x_data, y_data, epochs = runs, verbose = 1)
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
	# Display the result
	# --------------------------------------------------------------------------
	# Input:
		# x_data
		# y_data
		# runs
	# --------------------------------------------------------------------------
	# Output:
	# --------------------------------------------------------------------------
	def plot(self, x_data, y_data, y_predicted):
		plt.scatter(x_data[::1], y_data[::1], s = 4)
		plt.plot(x_data, y_predicted, 'r', linewidth = .4)
		plt.show()
	# --------------------------------------------------------------------------
## ------------------------------------------------------------------------------



# Use a probabilistic neural network (single layer) (MDN)

# Our MDN in PyTorch uses an Adam optimizer with
# a learning rate of 0.001 (the default), with a hyperbolic
# tangent as the activation function. The weights of the
# NN are initialized randomly. The MDN was found to
# be converged after training for 100,000 epochs, which is
# used throughout this work.


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
	nn.plot(x_data, y_data, y_pred)
# -------------------------------------------------------------------------------