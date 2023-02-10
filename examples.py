import sys

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
			nn = ml.NeuralNetwork(network, input_l, output_l, num_hl, hid_l)
			# Display the model summary 
			nn.display()
			# Train the model
			nn.fit(x_data, y_data, 5000, 0)
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
# Example problem fitting data for a non-linear sine function
# FROM: https://lucidar.me/en/neural-networks/curve-fitting-nonlinear-regression/
	  # f(x) = .1*x*cos(x) + .1*constant
# --------------------------------------------------------------------------------
# Input:		 
# --------------------------------------------------------------------------------
# Output:
# --------------------------------------------------------------------------------
def example_prob():
	
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
# --------------------------------------------------------------------------------