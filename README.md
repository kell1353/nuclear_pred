# Overview
This program provides a look into the application of neural networks to predict values of interest in nuclear physics. The idea of using a simple neural network to predict the mass excess for nuclei not yet experimentally measured is cited here:

Lovell, A. E., Mohan, A. T., Sprouse, T. M., and M. R. Mumpower. 
"Nuclear masses learned from a probabilistic neural network." 
ArXiv, (2022). 
https://doi.org/10.1103/PhysRevC.106.014305.

The comparison data used comes from here:

Moller, P., Sierk, A. J., Ichikawa, T., and H. Sagawa. 
"Nuclear ground-state masses and deformations: FRDM(2012)." ArXiv, (2015). 
https://doi.org/10.1016/j.adt.2015.10.002.

I came across the former paper sometime in early spring of 2022 while taking nuclear physics as a graduate student at SDSU. I found the application of neural networks to physics interesting at the time, but could not test it due to my own research work. Now as I have completed my Master's thesis, I am able to test it myself. 

This project is inspired by the ideas in this paper and the code is written completely by me. The premise: by changing only the input features and keeping the neural network simple and static, predictive performance of the model would get better. They showed a decrease to the MSE when compared to the FRDM results for nuclei outside the training set.  

During this project I was able gain a deeper understanding for neural networks both conceptually and programmatically with the help of the TensorFlow and PyTorch docs. This code contains the necessary files and modules to verify the results referred to above! 


# Modules (main features)
## main.py
Used to test any of the networks by choosing different model spaces and various other parameters
- type of architecure: PyTorch ['pyt'] or TensorFlow ['tf']
- type of network: Mixture Density Network ['mdn'] or Neural Network ['nn']

## testing.py
- Contains functions to test the various neural network classes against common examples
- Also contains the function used to verify the results from the paper (NOTE: still working out some kinks for the mdn)

## ML_lib.py
Contains three seperate neural network classes 
- First: Initializes an arbitrary NN in TensorFlow using Keras
- Second: Initializes an arbitrary NN in PyTorch with a single hidden layer 
- Third: Initializes an mixture density network in PyTorch with a single hidden layer using mdn.py

## mdn.py
Contains all the necessary code to evaluate a mixture density network 
- The network class, functions to compute a gaussian distribution and negative likelihood loss, and sample from the mixture

## plot.py
Contains all of our plotting functions for specific cases
- plots: comparitive heatmap between AME and FRDM | loss per epoch | probability density

## setup.py
Contains all the functions required to setup the full dataframes and chose/view the model input space from the AME and FRDM data.
- functions: read | snapshot | setup_full_data | model_select

## features.py
Contains functions that calculate the observables used in the input space. Higher models contain all previous parameters.

#### Standard Space 

(Model 2):
- Number of neutrons (N)
- Number of protons (Z)

#### Liquid Drop Model parameters

(Model 6):
- Volume term (A) 
- Surface term 
- Coulomb term 
- Asymmetry term

(model 8):
- Pairing terms (for protons and neutrons seperately)

#### Nuclear Shell model parameters

(model 10):
- Compares the number of nucleons to each known experimental magic number and returns the value with the number of protons(neutrons) on top of a magic number or away from a magic number (whichever is lower) (for protons and neutrons seperately)

(model 12): 
- Getting the shell of the last protons(neutron) for each nuclei (for protons and neutrons seperately)

# Results
I verified the results using a mixture density network and using a standard neural network architecture as well! The .pdf files are located in the output directory. 
- model_M[model_num]_nn - corresponds to the output of a standard neural network 
- model_M[model_num]_mdn - corresponds to the output of the mixture density network 

