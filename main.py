import sys
import numpy as np

import ML_lib as ml
import setup as su
import plot as pl


# Set the path for the training and testing files. Then read them into
# a panda dataframe
# --------------------------------------------------------------------
training_data = 'training_data.txt'
testing_data = 'testing_data.txt'

# Read in the AME data from the text file
AME_data = su.read_data(4, training_data)

# Read in the FRDM data from the text file
FRDM_data = su.read_data(4, testing_data)
# --------------------------------------------------------------------



# Create the full dataframe used for the model training data
# --------------------------------------------------------------------
res = su.setup_full_data(AME_data)

# get number of rows and columns from the dataframe
n_rows, n_cols = res[0], res[1]
# return the full datafram
full_data = res[2]
# --------------------------------------------------------------------



# gives a snapshot of the full dataframes
# --------------------------------------------------------------------
su.snapshot(20, full_data)
print('')
su.snapshot(20, FRDM_data)
# --------------------------------------------------------------------



# SAMPLE NUCLEI DATA FOR THE TRIANING SET
# selecting the model that we will use for the training data (human input)
# --------------------------------------------------------------------
M2 = su.model_select(1, full_data) 	# only proton and neutron data
# M6 = model_select(2, AME_data) 	# contains some semi-empirical mass formula terms
# M8 = model_select(3, AME_data) 	# contains protons/neutrons pairing data
# M10 = model_select(4, AME_data) 	# contains proximity to the magic numbers
# M12 = model_select(5, AME_data) 	# contains proton/neutrons shell data

# Convert dataframe values to numpy arrays
x_train, y_train = M2[0].to_numpy(), M2[1].to_numpy()

# Normalize the output training data 
# y_train = ml.minmax_norm(y_train)

# Setup the network
# -----------------------------------------------------
# number of hidden layers
num_hl = 1 	
# num. of units in input, output and hidden layers
input_l, output_l, hid_l = 2, 1, [60, 60] 

# Creates the neural network class
nn = ml.NeuralNetwork(input_l, output_l, num_hl, hid_l)
# Display the model summary 
nn.display()
# Train the model
nn.fit(x_train, y_train, 1000)
# -----------------------------------------------------
# --------------------------------------------------------------------



# using the model to calculate the mass excess for all the available N, Z
# in the FRDM (testing) dataset
# --------------------------------------------------------------------
# Setup the test array from the FRDM data
x_test = FRDM_data[['N', 'Z']].to_numpy()
y_test = FRDM_data['ME_F'].to_numpy()

# Test the model
y_pred = nn.predict(x_test)
# --------------------------------------------------------------------
# ml.example_prob()


frdm_ME = pl.create_matrix(FRDM_data['Z'], FRDM_data['N'], y_test)
model_ME = pl.create_matrix(FRDM_data['Z'], FRDM_data['N'], y_pred)


# Plot the output 
# --------------------------------------------------------------------
# exp_ME = su.exp_result(n_rows, [120, 170], AME_data.N, AME_data.Z, \
					   # AME_data.MASS_EXCESS)
# mod_ME = 

pl.plot_output(abs(frdm_ME - model_ME))
# --------------------------------------------------------------------