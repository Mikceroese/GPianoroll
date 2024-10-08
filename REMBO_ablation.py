#################################################
# Test routine for a Random Embedding Bayesian  #
# Optimization ablation study.                	#
#                                               #
# The program loads a model trained over scores #
# for a few epochs and then tries different     #
# dimensionality reduction configurations for   #
# the optimization. The objective function is   #
# the assymetric loss between a randomly        #
# (high dimensional) sampled song and the       #
# BO obtained samples.                          #               
#                                               #
# Author: Miguel Marcos				            #
#################################################

# Import zone

from os.path import join
from os import getcwd

import math
import numpy as np
import json
from datetime import datetime

import torch

from bayesoptmodule import BayesOptContinuous

from musegan import Generator # The model is defined here
from musegan import clip_samples

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
params['verbose_level'] = 0 # Quiet

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Class definitions

class AssymetricLoss(torch.nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
      super(AssymetricLoss, self).__init__()

      self.gamma_neg = gamma_neg
      self.gamma_pos = gamma_pos
      self.clip = clip
      self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
      self.eps = eps

    def forward(self, x, y):
      """
      Parameters
      ----------
      x: input logits
      y: targets
      """

      # Calculating probabilities
      x_sigmoid = torch.sigmoid(x)
      xs_pos = x_sigmoid
      xs_neg = 1 - x_sigmoid

      # Asymmetric Clipping
      if self.clip is not None and self.clip > 0:
          xs_neg = (xs_neg + self.clip).clamp(max=1)

      # Basic CE calculation
      los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
      los_neg = (1-y) * torch.log(xs_neg.clamp(min=self.eps))
      loss = los_pos + los_neg

      # Asymmetric Focusing
      if self.gamma_neg > 0 or self.gamma_pos > 0:
        if self.disable_torch_grad_focal_loss:
          torch.set_grad_enabled(False)
        pt0 = xs_pos * y
        pt1 = xs_neg * (1 - y) # pt = p if t > 0 else 1 - p
        pt = pt0 + pt1
        one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
        one_sided_w = torch.pow(1 - pt, one_sided_gamma)
        if self.disable_torch_grad_focal_loss:
          torch.set_grad_enabled(True)
        loss *= one_sided_w

      return -loss.sum([1,2,3]).mean()

class BayesOptMuseGAN(BayesOptContinuous):

    def __init__(self, rembo_dim, target, clipping = False):
       
        super().__init__(rembo_dim)

        self.n = rembo_dim
        self.mat_A = np.random.randn(self.n,true_dim)
        # Precalculate variances for remapping
        # (See 'remap_query' below)
        self.vars = np.sum(np.square(self.mat_A),axis=0)
        self.stds = np.sqrt(self.vars)

        self.loss = AssymetricLoss(gamma_neg=16,gamma_pos=4,disable_torch_grad_focal_loss=True)
        if torch.cuda.is_available():
            self.loss = self.loss.cuda()
        self.target_sample = target
        self.clipping = clipping

        self.min_score = 0.5
        self.max_score = 12

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
            score = self.loss(sample,self.target_sample)
            score = (score-self.min_score)/(self.max_score-self.min_score)

        return score    

# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Parameter grid search

n_targets = 5
rembo_dim = [4,10,20,30]
clipping = [True, False]
n_initial_samples = [8,16,32]
n_iter = [8,16,32,64]
rembo_n = 3

results = {}

total_start = datetime.now()

for t in range(n_targets):

    q = np.random.randn(true_dim)
    q = np_to_tensor(q)
    if torch.cuda.is_available():
        q = q.cuda()
    target_sample = gen(q)
    target_sample = clip_samples(target_sample,note_thresholds)

    results[t] = {}

    for dim in rembo_dim:
        results[t][dim] = {}
        lb = np.zeros((dim,)) # Lower bounds
        ub = np.ones((dim,)) # Upper bounds
        for clip in clipping:
            results[t][dim][clip] = {}
            for s in n_initial_samples:
                results[t][dim][clip][s] = {}
                params['n_init_samples'] = s
                for it in n_iter:
                    results[t][dim][clip][s][it] = []
                    params['n_iterations'] = it
                    
                    print("Target",t)
                    print("Dim",dim)
                    print("Clipping:",clip)
                    print(s,"initial samples,",it,"iterations.")

                    for a in range(rembo_n):

                        print("Embedding nº",a)

                        start = datetime.now()

                        bo = BayesOptMuseGAN(dim,target_sample,clipping=clip)
                        bo.params = params
                        bo.upper_bound = ub
                        bo.lower_bound = lb

                        mvalue, x_out, error = bo.optimize()

                        if error:
                            print("Something went wrong,skipping")
                            continue

                        stop = datetime.now()
                        time = (stop-start)
                        time = time.total_seconds()

                        print("Done.")

                        mvalue = mvalue*(bo.max_score-bo.min_score)+bo.min_score

                        results[t][dim][clip][s][it].append({"time":time,"mvalue":mvalue})

                    with open("results.json","w") as fp:
                        json.dump(results,fp,indent=4)
                
                    print("----------")

total_time = datetime.now() - total_start

print("Dumped results into \"results.json\"")
print("Total time:",total_time.total_seconds())