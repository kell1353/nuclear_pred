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
# su.snapshot(20, full_data)
# print('')
# su.snapshot(20, FRDM_data)
# print('')
# --------------------------------------------------------------------




# selecting the model that we will use for the training data 
# then constuct the model and train it against the testing data
# --------------------------------------------------------------------
# input for model selection
print('')
print('The model data used in calculating the mass excess:')
print('-----------------------------------------------------------------')
print('Enter 2, for Model 1:') 
print('- contains the highest level data. Number of' \
	    ' protons and neutrons in nuclei \n')
print('Enter 6, for Model 2:')
print('- adds the first four terms from the liquid drop model \n')
print('Enter 8, for Model 3:')
print('- adds a pairing term for both protons and neutrons \n')
print('Enter 10, for Model 4:') 
print('- adds the proximity of the last nucleon' \
	    ' to a magic number for both protons and neutrons \n')
print('Enter 12, for Model 5:')
print('- adds two terms characterizing the' \
	    ' nuclear shell of the last nucleon for both protons and neutrons')
print('-----------------------------------------------------------------')

print('')
model_num = int(input('Please enter the integer corresponding to the ' \
				  'model you wish to use: '))
print('')

# get model data
M = su.model_select(model_num, full_data) 

# Convert dataframe values to numpy arrays
x_data, y_data = M[0].to_numpy(), M[1].to_numpy()


# Setup the network 
# -----------------------------------------------------
# number of hidden layers
num_hl = 1
# num. of units in input, output and hidden layers
input_l, output_l, hid_l = model_num, 1, [6] 

# Creates the neural network class
nn = ml.NeuralNetwork('mse', input_l, output_l, num_hl, hid_l)
# Display the model summary 
nn.display()
# Train the model
nn.fit(x_data, y_data, 5000, 0)
# Plot loss over epochs
nn.plot_loss()
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
pl.plot_output(del_M)
# -----------------------------------------------------
# --------------------------------------------------------------------




# MSE BETWEEN MODEL AND FRDM FOR NON-EXPERIMENTAL NUCLEI
# --------------------------------------------------------------------
# Setup the test array from the FRDM data
# x_test = FRDM_data[['N', 'Z']].to_numpy()
# y_test = FRDM_data['ME_F'].to_numpy()

# Evaluate the model using the 
# mod_test = nn.evaluate(x_test, y_test)

# Use the model on the nuclei NOT included in the AME_data and compare 
# to the FDRM values
# ----------------------------------------------------------------
# y_pred = nn.predict(x_test)
# ----------------------------------------------------------------
# --------------------------------------------------------------------
