import matplotlib.pyplot as plt
import numpy as np

import random

## ------------------------------------------------------------------------------
# Creates a matrix where # of neutrons and # protons correspond to the indices
# -------------------------------------------------------------------------------
# Input:
	# Z 			array 		proton data from the data frame
	# N 			array 		neutron data from the data frame
	# del_ME 		matrix 		contains the difference from the data
# -------------------------------------------------------------------------------
# Output:
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
# the values correspond to the comparison of the model ME to the FRDM data
# -------------------------------------------------------------------------------
# Input:
	# mod_mat
	# frdm_mat
# -------------------------------------------------------------------------------
# Output:
	# del_ME
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
	# del_ME 		matrix 		contains the difference from the FDRM
# -------------------------------------------------------------------------------
# Output:
# -------------------------------------------------------------------------------

def plot_output(del_ME):
	fig, ((ax1)) = plt.subplots(nrows=1, ncols=1)

	# len_array = len(N)

	# Constructs a heat map using the martix data
	# ------------------------------------
	im = plt.imshow(del_ME, cmap='hot_r', interpolation='nearest', origin='lower', vmin=0, vmax=6)
	# ------------------------------------

	# Plot settings 
	# ------------------------------------
	# set labels
	ax1.set_xlabel('N', size = 12)
	ax1.set_ylabel('Z', size = 12)

	# remove spines
	ax1.spines['top'].set_visible(False)
	ax1.spines['right'].set_visible(False)

	# color bar settings
	cbar = plt.colorbar(im, fraction=0.033, pad=0.04)
	cbar.set_ticks([1, 2, 3, 4, 5, 6])
	cbar.set_label("Mass Difference (MeV) $[\u03B4_{ME}]$", size = 12)
	# ------------------------------------
	

	plt.show()
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
