"""A module for a mixture density network layer
For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import torch
import torch.nn as nn

import numpy as np
import sys

# normalization factor for Gaussians
oneDivSqrtTwoPI = 1.0 / np.sqrt(2 * np.pi)

# https://notebook.community/hardmaru/pytorch_notebooks/mixture_density_networks
class MDN(nn.Module):

    def __init__(self, n_inputs, n_hidden, n_gaussians):
        super(MDN, self).__init__()
        self.z_h = nn.Sequential(
                   nn.Linear(n_inputs, n_hidden),
                   nn.Tanh()
        )
        self.z_pi = nn.Linear(n_hidden, n_gaussians)
        self.z_sigma = nn.Linear(n_hidden, n_gaussians)
        self.z_mu = nn.Linear(n_hidden, n_gaussians)  

    def forward(self, x):
        z_h = self.z_h(x)
        pi = nn.functional.softmax(self.z_pi(z_h), -1) # USE log_softmax
        sigma = torch.exp(self.z_sigma(z_h))
        mu = self.z_mu(z_h)
        return pi, sigma, mu


def gaussian_distribution(y, mu, sigma):
    # make |mu|=K copies of y, subtract mu, divide by sigma
    result = (y.expand_as(mu) - mu) * torch.reciprocal(sigma)
    result = -0.5 * (result * result)
    return (torch.exp(result) * torch.reciprocal(sigma)) * oneDivSqrtTwoPI



# avoids problems of numerical instability
def log_gaussian_probability(y, mu, sigma):
    # ensuring the target is in correct dimensions
    y = y.expand_as(mu)
    # normalization constant
    constant = torch.add(-torch.log(sigma), - 0.5 * np.log(2*np.pi)) 
    # the values within the expnential
    result = (y - mu) * torch.reciprocal(sigma)
    norm = -0.5 * (result * result)

    return torch.add(constant, norm)

def mdn_loss(pi, sigma, mu, y):
    # result = gaussian_distribution(y, mu, sigma) * pi
    # result = torch.sum(result, dim=1)
    # result = -torch.log(result)
    log_prob = torch.log(pi) + log_gaussian_probability(y, mu, sigma)
    nll = -torch.logsumexp(log_prob, dim=1)
    return torch.mean(nll)



def gumbel_sample(x, axis=1):
    z = np.random.gumbel(loc=0, scale=1, size=x.shape)
    return (np.log(x) + z).argmax(axis=axis)


# choose the most relavent Gaussian for the data
def choose_gaussian(num_gaussians, mus, pis):

    pi_avgs = np.zeros(num_gaussians)

    for i in range(num_gaussians):
        pi_avgs[i] = np.average(pis[:, i])

    # choose the Gaussian with the highest average value of pi
    max_val, max_ind = max(pi_avgs), np.argmax(pi_avgs)

    return max_ind, mus[:, max_ind]

    



# TESTING: Currently testing this function
def best_mean(num_gaussians, y, mus):

    y_pred = np.zeros(len(y))

    for i in range(len(y)):
        min_val = 1000

        for j in range(num_gaussians):
            res = abs(y[i] - mus[i, j])

            if (res < min_val): 
                min_val = res
                y_pred[i] = mus[i, j] 

    return y_pred







