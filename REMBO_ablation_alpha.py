#################################################
# Test routine for a Random Embedding Bayesian  #
# Optimization ablation study.                	#               
#                                               #
# Author: Miguel Marcos				            #
#################################################

# Import zone

from os.path import join
from os import getcwd

import math
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import scipy.stats as sp

from bayesoptmodule import BayesOptContinuous

from musegan import Generator # The model is defined here
from musegan import clip_samples
from musegan import used_pitch_classes, tonal_distance

from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim

from utils import *

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Model instantiation

wd = getcwd() # Working directory
cp_path_gen = join(wd,'museGANgen_DBG_chroma_256_25k.pt') # Path to the generator checkpoint
gen = Generator()
load_checkpoint(cp_path_gen,gen)
if torch.cuda.is_available():
    gen = gen.cuda()
gen.eval()

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Fixed parameters

note_thresholds = [0.60828567, 0.55597573, 0.54794814]
true_dim = 256 # True dimension of the model's input

params = {}
params['init_method'] = 2 # Sobol
params['n_iter_relearn'] = 1
params['l_type'] = 'mcmc'
params['verbose_level'] = 5 # Log

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Extract 

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Class definitions

class BayesOptMuseGAN(BayesOptContinuous):

    def __init__(self, rembo_dim, target, clipping = False):
       
        super().__init__(rembo_dim)

        self.n = rembo_dim
        self.mat_A = np.random.randn(self.n,true_dim)
        # Precalculate variances for remapping
        # (See 'remap_query' below)
        self.vars = np.sum(np.square(self.mat_A),axis=0)
        self.stds = np.sqrt(self.vars)

        self.target_sample = target
        self.target_np = self.target_sample.cpu().detach().numpy()
        self.clipping = clipping

        self.upcs = used_pitch_classes(self.target_np,verbose=False)
        self.td = tonal_distance(self.target_np,verbose=False)
        self.sr = 1-np.mean(self.target_np,axis=(2,3)).squeeze()
        self.target_features = np.concatenate([self.upcs,self.td,self.sr])
        
        self.data_avgs = np.array([3.5037, 2.3047, 0.9134, 0.9825, 0.9646, 0.9832])
        self.data_stds = np.array([1.9138, 0.8805, 0.2439, 0.0083, 0.0240, 0.0046])

        self.target_features -= self.data_avgs
        self.target_features /= self.data_stds

        self.min_score = 0.0
        self.max_score = 1.0
        self.best_p = 100000.0
        self.best_points=[]

    def uniform_to_normal(self,query):

        # IMPORTANT: Box-Muller needs an even number of variables
        # If the program outputs an index-out-of-bounds error, it's
        # probably because of this.

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

        # Map query to the high-dimensional space
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

        q = self.uniform_to_normal(query)
        q = self.remap_query(q)
        q = np_to_tensor(q)
        if torch.cuda.is_available():
            q = q.cuda()
        sample = gen(q)
        if self.clipping:
            sample = clip_samples(sample,note_thresholds)
        return sample

    def evaluateSample(self,query):

        score = None

        with torch.no_grad():

            sample = self.generate_sample(query)

            # Use this instead of ssim if using musical features
            sample_np = sample.cpu().detach().numpy()
            upcs = used_pitch_classes(sample_np,verbose=False)
            td = tonal_distance(sample_np,verbose=False)
            sr = 1 - np.mean(sample_np,axis=(2,3)).squeeze()
            sample_features = np.concatenate([upcs,td,sr])

            sample_features -= self.data_avgs
            sample_features /= self.data_stds
            score = np.linalg.norm(sample_features-self.target_features)

            if self.best_p > score:
                self.best_p = score
            
            self.best_points = np.append(self.best_points,self.best_p)

        return score    

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Parameter grid search

n_targets = 3
colors = ['r','g','b']
rembo_dim = 10
clipping = True
init_s = 10
budget = 90
rembo_n = 5

np.random.seed(42)
qs = np.random.randn(n_targets,true_dim)

total_start = datetime.now()

fig = plt.figure()

for t in range(n_targets):

    q = qs[t]
    q = np_to_tensor(q)
    if torch.cuda.is_available():
        q = q.cuda()
    target_sample = gen(q)
    target_sample = clip_samples(target_sample,note_thresholds)

    lb = np.zeros((rembo_dim,)) # Lower bounds
    ub = np.ones((rembo_dim,)) # Upper bounds

    params['n_init_samples'] = init_s
    params['n_iterations'] = budget
    
    print("Target",t)

    np.random.seed(seed=42)

    full_results = []

    for a in range(rembo_n):

        print("Embedding nº",a)

        params['log_filename'] = "bopt_log"+str(a)+".txt" 

        bo = BayesOptMuseGAN(rembo_dim,target_sample,clipping=clipping)

        bo.params = params
        bo.upper_bound = ub
        bo.lower_bound = lb

        mvalue, x_out, error = bo.optimize()
        full_results.append(bo.best_points)

        if error:
            print("Something went wrong,skipping")
            continue

        print("Done.")

    res = np.asarray(full_results)
    res_mean = np.mean(res,axis=0)
    res_std = np.std(res,axis=0)
    
    it = range(res_mean.shape[0])
    t_limits = sp.t.interval(0.95,rembo_n) / np.sqrt(rembo_n)

    color = colors[t]
    symbol = ','
    label = 'Target_'+str(t)

    plt.plot(it,res_mean, color+symbol+'-', linewidth=2, label=label)
    plt.fill(np.concatenate([it,it[::-1]]),
             np.concatenate([res_mean + t_limits[0] * res_std,
                             (res_mean + t_limits[1] * res_std)[::-1]]),
             alpha=.3, fc=color, ec='None')
    plt.draw()

    print("----------")

plt.show(block=True)