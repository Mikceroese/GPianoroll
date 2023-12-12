#################################################
# Test program for using the bayesopt interface #
#                                               #
# The program loads a VAE trained over scores   #
# for a few epochs and then tries to create one #
# that best fits the human feedback.            #
#                                               #
# Since we are trying to find a maximum, the    #
# test function returns a negative value.       #
#################################################

# Import zone

from os.path import join
from os import getcwd

import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

import bayesopt
from bayesoptmodule import BayesOptContinuous

from museGAN import Generator, Discriminator # The model is defined here
from museGAN import clip_samples, samples_to_multitrack, write_sample

from utils import *

# Python3 compat
if hasattr(__builtins__, 'raw_input'):
    input = raw_input

wd = getcwd() # Working directory
sf2_path = join(wd,"SGM.sf2") # Soundfont file, needed to create .wav files
cp_path_gen = join(wd,'museGANgen_DBG_chroma_256_25k.pt') # Path to the generator checkpoint
cp_path_disc = join(wd,'museGANdisc_DBG_chroma_256_25k.pt')

sample_name = join(wd,'sample')
wav_name = sample_name+'.wav'
target_sample_name = join(wd,'target_sample')
target_wav_name = target_sample_name+'.wav'

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Model instantiation

gen = Generator()
load_checkpoint(cp_path_gen,gen)
gen.eval()
disc = Discriminator()
load_checkpoint(cp_path_disc,disc)
disc.eval()

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Parameters

note_thresholds = [0.60828567, 0.55597573, 0.54794814]

params = {}
params['n_init_samples'] = 64 # Probably use a small number if you plan to use human feedback
params['noise'] = 1e-6 # Default 1e-6, if human feedback is used, pump it up
params['n_iterations'] = 64
params['n_iter_relearn'] = 1
params['l_type'] = 'mcmc'
params['init_method'] = 2 # Sobol
params['verbose_level'] = 0

n = 10 # Problem is 10-dimensional
true_n = 256 # True dimension of the model's imput
lb = np.zeros((n,)) # Lower bounds
ub = np.ones((n,)) # Upper bounds

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Problem definition

class BayesOptMuseGAN(BayesOptContinuous):

    def __init__(self, n, mat_A, target_query):

        super().__init__(n)

        # REMBO matrix
        self.mat_A = mat_A
        # Problem dimension
        self.n = n

        # Precalculate variances for remapping
        # (See 'remap_query' below)
        self.vars = np.sum(np.square(mat_A),axis=0)
        self.stds = np.sqrt(self.vars)

        # Target sample
        self.target = self.generate_sample(target_query)
        # Target sample's features
        _ , self.target_f = disc(self.target)

    def uniform_to_normal(self,query):

        # Box-Muller transform

        even = np.arange(0,query.shape[-1],2)
        q_even = query[even]
        q_even[q_even==0] += 1e-6
        Rs = np.sqrt(-2*np.log(q_even))
        thetas = 2*math.pi*(query[even+1])
        cos = np.cos(thetas)
        sin = np.sin(thetas)
        query = np.stack([Rs*cos,Rs*sin],-1).flatten()

        return query

    def remap_query(self,query):

        # Map query to the high-dimension space
        q = np.matmul(query,self.mat_A)

        # Queries are mapped to a ~N(0,1) distribution
        # using the Box-Muller transform.
        # Therefore, the vector-matrix product (q)
        # results in a vector of sums of Normal distributions,
        # which are Normal themselves.

        # Each component i of the resulting vector will follow a Normal
        # distribution with mean 0 and variance V[i] equal
        # to the sum of the squares of the i-th column of mat_A.

        # Thus, we can remap each of this components to ~N(0,1)
        # using the precalculated standard deviations:

        q = np.divide(q,self.stds)

        # Clip the resulting query
        q = np.clip(q,-4.0,4.0)

        return q

    def generate_sample(self,query):

        self.uniform_to_normal(query)
        q = self.remap_query(query)
        sample = gen(np_to_tensor(q))
        sample = clip_samples(sample,note_thresholds)
        return sample

    def evaluateSample(self,query):

        with torch.no_grad():
            sample = self.generate_sample(query)
            _, sample_f = disc(sample)

        score = torch.nn.functional.l1_loss(sample_f,self.target_f,reduction='sum')

        return score

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Optimization

mat_A = np.random.randn(n,true_n)
target_q = np.random.uniform(0,1.0,n)
print("TARGET QUERY:", target_q)

bo = BayesOptMuseGAN(n, mat_A, target_q)

bo.params = params
bo.lower_bound = lb 
bo.upper_bound = ub

mvalue, x_out, error = bo.optimize()
print("Result", mvalue, "at", x_out)

with torch.no_grad():

    sample = bo.generate_sample(x_out)
    sample = clip_samples(sample,note_thresholds)
    m = samples_to_multitrack(tensor_to_np(sample))
    write_sample(m,sf2_path,sample_name)

    target_sample = bo.generate_sample(target_q)
    target_sample = clip_samples(target_sample,note_thresholds)
    m = samples_to_multitrack(tensor_to_np(target_sample))
    write_sample(m,sf2_path,target_sample_name)
