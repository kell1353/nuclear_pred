import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np


## ------------------------------------------------------------------------------
# Creates a matrix where # of neutrons and # protons correspond to the indices
# -------------------------------------------------------------------------------
# Input:
	# Z 			array 		proton data from the data frame
	# N 			array 		neutron data from the data frame
	# del_ME 		matrix 		contains the difference from the data
# -------------------------------------------------------------------------------
# Output:
	# a 			matrix
# -------------------------------------------------------------------------------
def create_matrix(Z, N, del_ME):

	# Creates a matrix based on the proton neutron values
	# ------------------------------------	
	a = np.zeros((max(Z)+1, max(N)+1))
	# ------------------------------------

	# Sets a value for each Z, N corresponidng to a matrix element
	# ------------------------------------
	for i in range(len(Z)):
		a[Z[i], N[i]] = del_ME[i]
	# ------------------------------------

	return a
# -------------------------------------------------------------------------------


## ------------------------------------------------------------------------------
# Creates a matrix where # of neutrons and # protons correspond to the indices
# the values correspond to the comparison of the model ME to the FRDM data. ONLY
# for nuclei measured in AME 2016
# -------------------------------------------------------------------------------
# Input:
	# mod_mat		matrix 		contains the model mass excess data related to N, Z
	# frdm_mat		matrix 		contains the FRDM mass excess data related to N, Z
# -------------------------------------------------------------------------------
# Output:
	# del_ME 		matrix 		contains the absoulute comparison between two results
# -------------------------------------------------------------------------------
def matrix_sub(mod_mat, frdm_mat):

	mod_sh = mod_mat.shape
	nrows, ncols = mod_sh[0], mod_sh[1]

	del_ME = np.zeros((nrows+1, ncols+1))

	for i in range(nrows):
		for j in range(ncols):
			if mod_mat[i, j] != 0 and frdm_mat[i, j] != 0:
				del_ME[i, j] = abs(frdm_mat[i, j] - mod_mat[i, j])

	return del_ME
# -------------------------------------------------------------------------------




## ------------------------------------------------------------------------------
# Plots the difference matrix 
# -------------------------------------------------------------------------------
# Input:
	# title 		string 		title of the plot
	# filename 		string 	 	name of the output file
	# output_path	string 		path of the output directory  
	# del_ME 		matrix 		contains the difference from the FDRM
# -------------------------------------------------------------------------------
# Output:
# -------------------------------------------------------------------------------

def plot_output(title, filename, output_path, del_ME):
	fig, ((axes)) = plt.subplots(nrows=1, ncols=1)

	# Constructs a heat map using the martix data
	# ------------------------------------
	im = axes.imshow(del_ME, cmap='hot_r', interpolation='nearest', \
					 origin='lower', vmin=0, vmax=6)
	# ------------------------------------

	# Plot settings 
	# ------------------------------------
	# set labels
	axes.set_title(title)
	axes.set_xlabel('N', size = 12)
	axes.set_ylabel('Z', size = 12)

	# remove spines
	axes.spines['top'].set_visible(False)
	axes.spines['right'].set_visible(False)

	# color bar settings
	cbar = plt.colorbar(im, fraction=0.033, pad=0.04)
	cbar.set_label("Mass Difference (MeV) $[\u03B4_{ME}]$", size = 12)
	# ------------------------------------
	
	# Save plot to output folder
	fig.savefig(output_path + filename, dpi=300)

	print("")
	print("The heatmap has been saved to: " + output_path)
# -------------------------------------------------------------------------------



## ------------------------------------------------------------------------------
# Plots a scatter plot of the selected variable against the target quanity 
# -------------------------------------------------------------------------------
# Input:
	# var 			array 		variable data from the data frame
	# res 			array 		target result data from the data frame
	# labels 		array 		labels for the x, y axes
# -------------------------------------------------------------------------------
# Output:
# -------------------------------------------------------------------------------

def plot_var(var, res, labels):
	fig, ((ax1)) = plt.subplots(nrows=1, ncols=1)

	# Creates a scatter plot from the data
	# ------------------------------------
	ax1.scatter(var, res, s=6, c='k', marker='o')
	# ------------------------------------

	# Plot settings 
	# ------------------------------------
	# set labels
	ax1.set_xlabel(labels[0], size = 12)
	ax1.set_ylabel(labels[1], size = 12)

	# remove spines
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)
	# ------------------------------------
	
	plt.tight_layout()
	plt.show()
# -------------------------------------------------------------------------------



## ------------------------------------------------------------------------------
#  plot the density distribution of the target value x
# -------------------------------------------------------------------------------
# Input:
	# data 		array 			pandas.dataframe or numpy.ndarray
	# x 		array/string 	vectors or keys in data
	# labels 	array 			labels for the x, y axes
# -------------------------------------------------------------------------------
# Output:
# -------------------------------------------------------------------------------
def plot_density(data, x, labels):
	fig, ((ax)) = plt.subplots(nrows=1, ncols=1)

	sns.set_style('whitegrid')
	sns.kdeplot(data=data, x=x, fill = True, label = labels[0])

	# Plot settings 
	# ------------------------------------
	# set labels
	ax.set_xlabel("Mass Excess (MeV)", size = 12)
	ax.set_ylabel("Density", size = 12)

	# Turns off grid on the left Axis.
	ax.grid(False)
	# ------------------------------------

	plt.legend()
	plt.show()
# -------------------------------------------------------------------------------



# --------------------------------------------------------------------------
# Display the loss of training the model 
# --------------------------------------------------------------------------
# Input:
	# epochs		array 	number of iterations
	# loss 			array 	the values for the training loss per epoch
	# val_loss 		array 	the values for the testing loss per epoch
	# loss_type 	string  type of losse for the y axis label
# --------------------------------------------------------------------------
# Output:
# --------------------------------------------------------------------------
def plot_loss(epochs, losses, val_losses, loss_type):
	fig, ((ax1)) = plt.subplots(nrows=1, ncols=1)

	# plot training and testing loss
	# ------------------------------------
	ax1.plot(epochs, losses, 'r', c='b', linewidth=.4, \
			 label='training')
	ax1.plot(epochs, val_losses, 'r', c='k', linewidth=.4, \
			 label='validation')
	# ------------------------------------

	# Plot settings 
	# ------------------------------------
	# set labels
	ax1.set_xlabel('epochs', size = 12)
	ax1.set_ylabel(loss_type + ' loss', size = 12)

	# remove spines
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)

	# enable legend
	ax1.legend()
	# ------------------------------------

	plt.tight_layout()
	plt.show()
# --------------------------------------------------------------------------