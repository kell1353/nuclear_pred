import numpy as np
import pandas as pd
import sys

import features as f

 
## ------------------------------------------------------------------------------
# Reads in the data from the input file into a panda data frame
# -------------------------------------------------------------------------------
# Input:
	# num_skip	integer 		number of rows to skip at the beginning of the file
	# file 		string 			text file containg nucleon and mass data
	# dtypes 	dict 			associated columns with respective data types
# -------------------------------------------------------------------------------
# Output:
	# data 		data frame 		data frame containg the file data
# -------------------------------------------------------------------------------
def read_data(num_skip, file):
	data = pd.read_csv(file, delimiter="	", skiprows=num_skip)
	return data
## -------------------------------------------------------------------------------



## ------------------------------------------------------------------------------
# Access a sanpshot of the first n rows of the dataframe
# -------------------------------------------------------------------------------
# Input:
	# num_rows 		integer 		number of rows the snapshot shows
	# df 		 	dataframe 		the input dataframe 
# -------------------------------------------------------------------------------
# Output:
# -------------------------------------------------------------------------------
def snapshot(num_rows, df): 
	output = df.head(num_rows)

	print('')
	print(output)
# -------------------------------------------------------------------------------



## ------------------------------------------------------------------------------
# Reads in the data from the input file into a panda data frame
# -------------------------------------------------------------------------------
# Input:
	# d_name 	string 			name of data file (AME or FRDM)
	# data 		dataframe 		dataframe containing the proton & neutron data
# -------------------------------------------------------------------------------
# Output:
	# n_rows 		integer 		number of different nuclei in the data
	# n_cols 		integer 		number of dataframe columns
	# data 			dataframe 		dataframe containing additional model terms
# -------------------------------------------------------------------------------
def setup_full_data(d_name, data):

	# Get the shape of the data frame
	# -----------------------------------------------------------
	df_shape = data.shape

	# Get the number of (rows, columns)
	n_rows, n_cols = df_shape[0], df_shape[1]
	# -----------------------------------------------------------

	# convert mass excess to MeV (only for AME file)
	# -----------------------------------------------------------
	if d_name == "AME":
		data['MASS_EXCESS'] = data['MASS_EXCESS']*(10**(-3))
		data['ME_unc'] = data['ME_unc']*(10**(-3))
	# -----------------------------------------------------------


	# Variables for the liquad drop model 
	# -----------------------------------------------------------
	vol = f.get_volume_data(data.N, data.Z) 			# volume term
	surf = f.get_surface_data(vol)  							# surface term
	coul = f.get_coulomb_data(data.N, data.Z, vol) 		# Coulomb term
	asym = f.get_aymmetric_data(data.N, data.Z, vol)	# asymmetry term
								
	pair_p = f.get_pairing_data(n_rows, data.Z)				# pairing terms
	pair_n = f.get_pairing_data(n_rows, data.N)				
	# -----------------------------------------------------------


	# Variables for the nuclear-shell model
	# -----------------------------------------------------------
	# Neutron and Proton magic numbers
	N_mn, Z_mn = [2, 8, 20, 28, 50, 82, 126], [2, 8, 20, 28, 50, 82, 114]

	delta_Z = f.get_delta_data(n_rows, Z_mn, data.Z)
	delta_N = f.get_delta_data(n_rows, N_mn, data.N)
	Z_shell = f.get_shell_data(n_rows, data.Z)
	N_shell = f.get_shell_data(n_rows, data.N)
	# -----------------------------------------------------------


	# Add new columns to the data frame 
	# -----------------------------------------------------------
	data['LD1'] = vol
	data['LD2'] = surf
	data['LD3'] = coul
	data['LD4'] = asym
	data['LD5'] = pair_p
	data['LD6'] = pair_n

	data['SM1'] = delta_Z
	data['SM2'] = delta_N
	data['SM3'] = Z_shell
	data['SM4'] = N_shell
	# -----------------------------------------------------------

	return n_rows, n_cols, data
# -------------------------------------------------------------------------------




## ------------------------------------------------------------------------------
# Reads in the data from the input file into a panda data frame
# -------------------------------------------------------------------------------
# Input:
	# m_num 		integer 		the specific model designation number
	# full_data 	dataframe 		contains all of the possible model data
# -------------------------------------------------------------------------------
# Output:
	# M
# -------------------------------------------------------------------------------
def model_select(d_name, m_num, full_data):
	# Selecting the model space from the main data frame 
	# M2, M6, M10, M12,...
	# -----------------------------------------------------------------
	if (m_num == 2):
		# Model 1 contains the highest level data. Number of protons 
		# and neutrons in nuclei
		# ------------------------------------------
		M = full_data[['N', 'Z']]
		# ------------------------------------------

	# These models add data pertaining to the liquid drop model
	# ------------------------------------------
	elif (m_num == 6):
		# Model 2 adds the first four terms from the liquid drop model
		M = full_data[['N', 'Z', 'LD1', 'LD2', 'LD3', 'LD4']]

	elif (m_num == 8):
		# Model 3 adds a pairing term for both protons and neutrons
		M = full_data[['N', 'Z', 'LD1', 'LD2', 'LD3', 'LD4', 'LD5', 'LD6']]
	# ------------------------------------------

	# These models add data pertaining to the nuclear shell model
	# ------------------------------------------
	elif (m_num == 10): 
	# Model 4 adds the proximity of the last nucleon to a magic number for both 
	# protons and neutrons
		M = full_data[['N', 'Z', 'LD1', 'LD2', 'LD3', 'LD4', 'LD5', 'LD6', 'SM1', \
						 'SM2']]
	elif (m_num == 12):
	# Model 5 adds two terms characterizing the nuclear shell of the last nucleon 
	# for both protons and neutrons
		M = full_data[['N', 'Z', 'LD1', 'LD2', 'LD3', 'LD4', 'LD5', 'LD6', 'SM1', \
						 'SM2', 'SM3', 'SM4']]
	# ------------------------------------------

	else:
		print('')
		print('Please select a valid integer from: [2, 6, 8, 10, 12]')
		sys.exit()
	# -----------------------------------------------------------------

	if d_name == "AME": ME = full_data['MASS_EXCESS']
	elif d_name == "FRDM": ME = full_data['ME_F']

	return [M, ME]
# -------------------------------------------------------------------------------
