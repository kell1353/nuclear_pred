import numpy as np
import sys

## ------------------------------------------------------------------------------
# Constructing the volume term (A)
# 
# -------------------------------------------------------------------------------
# Input:
	# N 		array 		neutron data from the data frame
	# Z 		array 		proton data from the data frame
# -------------------------------------------------------------------------------
# Output:
	# vol 		array 		array containing the addition of protons and neutrons
# -------------------------------------------------------------------------------
def get_volume_data(N, Z):
	vol = N + Z
	return vol
# -------------------------------------------------------------------------------


## ------------------------------------------------------------------------------
# Constructing the surface term, 
# 
# -------------------------------------------------------------------------------
# Input:
	# A 		array
# -------------------------------------------------------------------------------
# Output:
	# surf 		array 		array containing the surface term for each nuclei
# -------------------------------------------------------------------------------
def get_surface_data(A):
	surf = A**(2/3)
	return surf
# -------------------------------------------------------------------------------


## ------------------------------------------------------------------------------
# Constructing the Coulomb term, 
# 
# -------------------------------------------------------------------------------
# Input:
	# N 		array 		neutron data from the data frame
	# Z 		array 		proton data from the data frame
	# A 		array
# -------------------------------------------------------------------------------
# Output:
	# coul 		array 		array containing the the Coulomb term for each nuclei
# -------------------------------------------------------------------------------
def get_coulomb_data(N, Z, A):
	coul = (Z*(Z - 1))/(A**(1/3))
	return coul
# -------------------------------------------------------------------------------


## ------------------------------------------------------------------------------
# Constructing the asymmetric term, 
# 
# -------------------------------------------------------------------------------
# Input:
	# N 		array 	neutron data from the data frame
	# Z 		array 	proton data from the data frame
	# A 		array
# -------------------------------------------------------------------------------
# Output:
	# asym 		array 	array containing the the asymmetric term for each nuclei
# -------------------------------------------------------------------------------
def get_aymmetric_data(N, Z, A):
	asym = ((N - Z)**2)/A
	return asym
# -------------------------------------------------------------------------------


## ------------------------------------------------------------------------------
# Constructing the pairing term, if the proton(neutron) value is even then leave 
# the value at 0, if it is odd set the value equal to 1
# -------------------------------------------------------------------------------
# Input:
	# len_array    integer 		number of nuclei in list
	# nucl_array   array     	proton(neutron) data 
# -------------------------------------------------------------------------------
# Output:
	# nucl_eo      array        pairing term containing even/odd data 
# -------------------------------------------------------------------------------
def get_pairing_data(len_array, nucl_array):

	nucl_eo = np.zeros(len_array, dtype=int)

	for i in range(len_array):
		# only change the value of the list if the number is odd 
		if (np.mod(nucl_array[i], 2) == 1): nucl_eo[i] = 1
	return nucl_eo
# -------------------------------------------------------------------------------







## ------------------------------------------------------------------------------
# Compares the number of nucleons to each known experimental magic number and 
# returns the value with the number of protons(neutrons) on top of a magic number 
# or away from a magic number (whichever is lower)
# -------------------------------------------------------------------------------
# Input:
	# len_array     integer 	number of nuclei in list
	# mag_nums 		array 		magic numbers for protons(neutrons)
	# nucl 			array 		number of protons(neutrons) for each nuclei
# -------------------------------------------------------------------------------
# Output:
	# deltas 		array 		magic number proximity value for each nuclei		
# -------------------------------------------------------------------------------
def get_delta_data(len_array, mag_nums, nucl):

	deltas = np.zeros(len_array, dtype=int)

	for i in range(len_array):
		# initially set a large value for comparison (gets overideded)
		top, delta = 1000, 1000

		for j in range(len(mag_nums)): 
			# calculate the top and delta values for each mag. num.
			tmp_top = nucl[i] - mag_nums[j]
			tmp_delta = abs(nucl[i] - mag_nums[j])

			# check if the current top & delta are the lowest non-negative
			# values respectively. 
			if (tmp_top < top) and (tmp_top >= 0): top = tmp_top
			if (tmp_delta < delta): delta = tmp_delta

		# set the result equal to the minimum of the two variables
		deltas[i] = min(top, delta)

	return deltas
# -------------------------------------------------------------------------------




## ------------------------------------------------------------------------------
# Getting the shell of the last protons(neutron) for each nuclei
# -------------------------------------------------------------------------------
# Input:
	# len_array 	integer 	number of nuclei in list
	# nucl 			array 		number of protons(neutrons) for each nuclei
# -------------------------------------------------------------------------------
# Output:
	# shell 		array 		shell number for the last nucleon for each nuclei
# -------------------------------------------------------------------------------
def get_shell_data(len_array, nucl):

	# the number of states (or number of nucleons that can exist in a shell) is 
	# proportial to the principal quantum number. In the harmonic oscillator 
	# basis this is N = 2n + l. n - nodal quantum number | l - orbital angular 
	# momentum, where the total angular momentum j = l +- 1/2.

	# using this setup the number of states is equivalent to the sum of all 
	# possible magnetic quantum numbers for each j. 

	# For example, the sd-shell consists of Od_5/2, 1s_1/2, and Od_3/2 orbitals
	# Corresponding m is then:
		# Od_5/2: m = -5/2, -3/2, -1/2, 1/2, 3/2, 5/2
		# 1s_1/2: m = -1/2, 1/2
		# Od_3/2: m = -3/2, -1/2, 1/2, 3/2
		# summed together gives us 12 states

	# Foetunately this model follows the triangular numbers
		# n(n + 1)/2 = 1, 3, 6, 10, 15, 21, 28, 36, 45,... only multiplied by 2

	shell = np.zeros(len_array, dtype=int)

	# search for the shell the proton(neutron) belongs in.
	for i in range(len_array):	

		if (nucl[i] == 0): shell[i] = 0

		else: 
			# counting up from 0 
			p_qn, tri_num = 0, 0

			# set the shell number once the nucleon number is less then 
			# the sum of twice the previous triangle number
			# print(nucl[i])
			while (nucl[i] > tri_num):
				p_qn += 1
				tri_num += p_qn*(p_qn + 1)

			shell[i] = p_qn - 1

	return shell
# -------------------------------------------------------------------------------