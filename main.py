import ML_lib as ml
import setup as su
import plot as pl

import testing as ts

import sys



# Set the path for the training and testing files. Then read them into
# a panda dataframe
# --------------------------------------------------------------------
# input_path
# output_path 

training_data = 'training_data.txt'
testing_data = 'testing_data.txt'

# Read in the AME data from the text file
AME_data = su.read_data(4, training_data)

# Read in the FRDM data from the text file
FRDM_data = su.read_data(4, testing_data)
# --------------------------------------------------------------------


# Create the full dataframe used for the model training data
# --------------------------------------------------------------------
res = su.setup_full_data("AME", AME_data)

# get number of rows and columns from the dataframe
n_rows, n_cols = res[0], res[1]
# return the full datafram
full_data = res[2]
# --------------------------------------------------------------------

# Create the full dataframe used for the model testing data
# --------------------------------------------------------------------
res = su.setup_full_data("FRDM", FRDM_data)

# return the full datafram
FRDM_full_data = res[2]
# --------------------------------------------------------------------


# gives a snapshot of the full dataframes
# --------------------------------------------------------------------
su.snapshot(20, full_data)
print('')
su.snapshot(20, FRDM_full_data)
print('')
# --------------------------------------------------------------------



# run testing functions if wanted 
# --------------------------------------------------------------------
# ts.nn_mdn_comparison(output_path, full_data, FRDM_full_data)
# sys.exit()
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

# get model training data
M = su.model_select("AME", model_num, full_data) 

# Convert dataframe values to numpy arrays
x_data, y_data = M[0].to_numpy(), M[1].to_numpy()



# Setup the network 
# -----------------------------------------------------
# type of architecure: PyTorch ['pyt'] or TensorFlow ['tf']
arch_type = "pyt"
# type of network: Mixture Density Network ['mdn'] or Neural Network ['nn']
network = "nn"  
title = "M"+str(model_num)+"_"+network

# number of hidden layers
num_hl = 1
# num. of units in input, output and hidden layers
input_l, output_l, hid_l = model_num, 1, [10] 


# Creates the neural network class using TensorFlow
if arch_type == 'tf':
	nn = ml.tf_NeuralNetwork(input_l, output_l, num_hl, hid_l)
# Creates the neural netowrk class using PyTorch
elif arch_type == 'pyt':
	# Creates the mixture density netowrk class using PyTorch
	if network == "mdn":
		num_gaussians = 4
		nn = ml.pyt_MixtureDensityNetwork(input_l, output_l, hid_l, num_gaussians)
	else:
		nn = ml.pyt_NeuralNetwork(network, input_l, output_l, hid_l)
else:
	print("")
	print("System Exit: Please use PyTorch ['pyt'] or TensorFlow ['tf']")
	sys.exit()

# Display the model summary 
if arch_type == 'tf':
	nn.display()
else:
	nn.display((0, model_num))

# Train the model
nn.fit(x_data, y_data, 20000, 500, 0)

# Save the model to .txt file
# nn.save(title+"_model", output_path)

# Plot loss or accuracy over epochs
if network == 'nn':
	pl.plot_loss(nn.epochs, nn.losses, nn.val_losses, 'MSE')
else:
	pl.plot_loss(nn.epochs, nn.losses, nn.val_losses, 'NLL')

# print('')
print("# of epochs: ", nn.epochs[-1])
print("Model loss: ", nn.losses[-1])
# -----------------------------------------------------
# --------------------------------------------------------------------




# using the model to calculate the mass excess for all the available 
# N, Z in the AME compared to the FDRM (testing) dataset (Fig. 1)
# --------------------------------------------------------------------
y_FRDM = FRDM_full_data['ME_F'].to_numpy()

# Use the model on the nuclei NOT included in the AME_data
# and compare to the FDRM values
# -----------------------------------------------------
if network == 'nn': 
	mod_pred = nn.predict(x_data) 
else: 
	pi_pred, sigma_pred, mu_pred = nn.predict(x_data)
	mod_pred = mu_pred
# -----------------------------------------------------

# setup the matrices for the datasets 
# -----------------------------------------------------
frdm_M = pl.create_matrix(FRDM_full_data['Z'], FRDM_full_data['N'], y_FRDM)
model_M = pl.create_matrix(full_data['Z'], full_data['N'], mod_pred)
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
M = su.model_select("FRDM", model_num, FRDM_full_data) 

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
# ----------------------------------------------------------------
# --------------------------------------------------------------------
