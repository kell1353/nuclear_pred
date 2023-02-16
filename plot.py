import matplotlib.pyplot as plt
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
