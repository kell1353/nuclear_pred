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

I came across the former paper sometime in early spring of 2022 while taking nuclear physics as a graduate student at SDSU. The premise: by changing only the input features and keeping the neural network simple and static, they showed an increased performance in predicting the mass excess for nuclei. I found the application of neural networks to physics interesting at the time, but could not test it out due to my own research work. Now as I have completed my Master's thesis, I am able to test it myself. This project is inspired by the ideas in this paper and the code is written completely by me. 

Because of this project I was able gain a deeper understanding for neural networks both concpetually and thorugh the TensorFlow and PyTorch docs programmatically as well. This code  contains the necessary files and modules to verify the results referred to above! 


# Modules
#### main.py

#### testing.py

#### ML_lib.py

#### mdn.py

#### plot.py

#### setup.py

#### features.py


# Results
I verified the results using a mixture density network and using a standard neural network architecture as well! The .pdf files are located in the output directory. 
- M[m_num]_nn - corresponds to the output of a standard neural network 
- M[m_num]_mdn - corresponds to the output of the mixture density network 

