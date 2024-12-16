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

import torch
import scipy.stats as sp

from bayesoptmodule import BayesOptContinuous
from bopt_dict import BayesOptDict

from musegan import Generator
from musegan import clip_samples, write_sample, samples_to_multitrack
from musegan import used_pitch_classes, tonal_distance

from itertools import combinations

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

# Class definitions

class AssymetricLoss(torch.nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
      super(AssymetricLoss, self).__init__()

      self.gamma_neg = gamma_neg
      self.gamma_pos = gamma_pos
      self.clip = clip
      self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
      self.eps = eps

      # For normalizaton
      self.zero = 0.444895
      self.one = 9.730950

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

      return (-loss.sum([1,2,3]).mean() - self.zero)/(self.one-self.zero)

class BayesOptMuseGAN(BayesOptContinuous):

    def __init__(self, rembo_dim, target, clipping = False, loss='MF'):
       
        super().__init__(rembo_dim)

        self.target_sample = target
        self.clipping = clipping

        self.best_asl = 100000.0
        self.asl_curve = []
        self.best_mf = 100000.0
        self.mf_curve = []

        self.target_np = self.target_sample.cpu().detach().numpy()
        self.upcs = used_pitch_classes(self.target_np,verbose=False)
        self.td = tonal_distance(self.target_np,verbose=False)
        self.sr = np.mean(self.target_np,axis=(0,2,3))
        self.target_features = np.concatenate([self.upcs,self.td,self.sr])

        self.n = rembo_dim
        self.mat_A = np.random.randn(self.n,true_dim)
        # Precalculate variances for remapping
        # (See 'remap_query' below)
        self.stds = np.sqrt(np.sum(np.square(self.mat_A),axis=0))
        self.scale = np.linalg.norm(self.stds)

        self.asl = AssymetricLoss(gamma_neg=16,gamma_pos=4,disable_torch_grad_focal_loss=True)
        if torch.cuda.is_available():
            self.asl = self.asl.cuda()

        self.loss = loss

        mid_q = np.ones(rembo_dim)
        self.mid_score = self.evaluateSample(mid_q)

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
        # using the precalculated variances:

        q = np.divide(q,self.scale)

        # Clip the resulting query
        q = np.clip(q,-4.0,4.0)

        return q

    def to_high_dim(self,query):

        q = self.uniform_to_normal(query)
        q = self.remap_query(q)
        q = np_to_tensor(q)
        return q

    def generate_sample(self,query):

        q = self.to_high_dim(query)
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
            mf_loss = self.compare_features(sample)
            if mf_loss < self.best_mf:
                self.best_mf = mf_loss
            asl_loss = self.asl(sample,self.target_sample).item()
            if asl_loss < self.best_asl:
                self.best_asl = asl_loss

            self.mf_curve.append(self.best_mf)
            self.asl_curve.append(self.best_asl)

            if self.loss=='MF':
                score = mf_loss
            else:
                score = asl_loss 

        return score    

    def compare_features(self,sample):
        sample_np = sample.cpu().detach().numpy()
        upcs = used_pitch_classes(sample_np,verbose=False)
        td = tonal_distance(sample_np,verbose=False)
        sr = np.mean(sample_np,axis=(0,2,3))
        sample_features = np.concatenate([upcs,td,sr])
        score = np.linalg.norm(sample_features-self.target_features)

        return score
        
# -.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*-.-*- #

# Parameters

loss_names = {"MF":"Music Features",
              "ASL":"Assymetric Loss"
              }

losses = ["MF",
          "ASL"
          ]

n_targets = 4
colors = ['r','g','b']
rembo_dim = 10
clipping = True
n_init = 9
extra_init = 0
n_samples = 56
rembo_n = 4 # Will test up to rembo_n-1
rembo_budgets = [(n_samples+n_init+1)//(n+1) for n in range(rembo_n-1)]

results = {}

note_thresholds = [0.60828567, 0.55597573, 0.54794814]
true_dim = 256 # True dimension of the model's input
save_midis = False

params = {}
params['init_method'] = 2 # Sobol
params['n_iter_relearn'] = 1
params['l_type'] = 'mcmc'
params['verbose_level'] = 0 # Quiet

params['n_init_samples'] = n_init
params['n_iterations'] = 0
params['load_save_flag'] = 2 # 1 - Load, 2 - Save, 3 - Load and save

params['save_filename'] = "musegan_bopt.txt"
params['verbose_level'] = 5
params['log_filename'] = "musegan_bopt_log.txt"

sf2_path = "SGM.sf2"

loss_mat = np.zeros((len(losses),rembo_n-1,len(losses)))

seed = 42
np.random.seed(seed)
qs = np.random.randn(n_targets,true_dim)

lb = np.zeros((rembo_dim,)) # Lower bounds
ub = np.ones((rembo_dim,)) # Upper bounds

fig = plt.figure()
subfigs = fig.subfigures(2,1)
axs = []
axs1 = subfigs[0].subplots(1,len(losses), sharey=True)
axs.append(axs1)
axs2 = subfigs[1].subplots(1,len(losses), sharey=True)
axs.append(axs2)
fig.suptitle("Historic best through optimization")
for i in range(len(losses)):
    axs[i][0].set_ylabel(losses[i])
for i in range(len(losses)):
    axs[0][i].set_title("Optimizing "+str(losses[i]))

for i_l, loss in enumerate(losses):
                
    print("---------------------------------------------------")
    print("Starting optimization with loss:",loss_names[loss])
    print("---------------------------------------------------")

    mf_curves = []
    asl_curves = []

    for t in range(n_targets):

        print("Target",t)

        q = qs[t]
        q = np_to_tensor(q)
        if torch.cuda.is_available():
            q = q.cuda()
        target_sample = gen(q)
        target_sample = clip_samples(target_sample,note_thresholds)

        if save_midis:
            target_name = "samples/target_"+str(t)
            m = samples_to_multitrack(tensor_to_np(target_sample))
            write_sample(m,sf2_path,target_name,write_wav=False)

        best_mvalue = 1000000.0

        for r in range(rembo_n):

            np.random.seed(seed*(r+1))

            print("Embedding nº",r+1)

            params['n_init_samples'] = n_init
            params['n_iterations'] = 0
            params['load_save_flag'] = 2 # 1 - Load, 2 - Save, 3 - Load and save

            bo = BayesOptMuseGAN(rembo_dim,target_sample,clipping=clipping,loss=loss)

            bo.params = params
            bo.upper_bound = ub
            bo.lower_bound = lb

            mvalue, x_out, error = bo.optimize()
            if error:
                print("Something went wrong,skipping")
                continue

            bo_dict = BayesOptDict(params['save_filename'])
            for i in range(extra_init):
                mid_sample = np.round(np.random.uniform(size=rembo_dim),3)
                mid_sample[0::2] = 1 - 1e-6
                bo_dict.add_sample(mid_sample,bo.mid_score)

            bo_dict.set_init_samples(n_init+extra_init)
            bo_dict.set_num_iter(n_samples)
            bo_dict.save_txt(params['save_filename'])

            params['n_init_samples'] = n_init + extra_init
            params['n_iterations'] = n_samples
            params['load_save_flag'] = 3 # 1 - Load, 2 - Save, 3 - Load and save
            params['load_filename'] = params['save_filename']

            bo.params = params

            mvalue, x_out, error = bo.optimize()
            if error:
                print("Something went wrong,skipping")
                continue

            if save_midis:
                if best_mvalue > mvalue:
                    best_mvalue = mvalue
                    sample = bo.generate_sample(x_out)
                    target_name = "samples/closest_"+str(t)+"_"+loss
                    m = samples_to_multitrack(tensor_to_np(sample))
                    write_sample(m,sf2_path,target_name,write_wav=False)

            mf_curves.append(bo.mf_curve)
            asl_curves.append(bo.asl_curve)

            print("Done.")

    for b_i, b in enumerate(rembo_budgets):

        color = colors[b_i]
        symbol = ','
        n_emb = b_i+1
        label = str(n_emb)+" embedding"
        if n_emb > 1:
            label = label+"s"

        res = np.asarray(mf_curves)
        if n_emb > 1:
            res = res[:,:b]
            res = np.reshape(res,(-1,rembo_n,b))
            rembo_res = []

            for r_i in range(res.shape[0]):
                r = res[r_i]
                cs = list(combinations(r,n_emb))
                cs_res = []
                for c in cs:
                    interleaved_c = np.empty(n_emb*b)
                    for i in range(n_emb):
                        interleaved_c[i::n_emb] = c[i]
                    for i in range(len(interleaved_c)-1):
                        if interleaved_c[i+1]>interleaved_c[i]:
                            interleaved_c[i+1] = interleaved_c[i]
                    cs_res.append(interleaved_c)
                chosen_cs = np.random.choice(len(cs),rembo_n,replace=False)
                rembo_res.append(np.asarray(cs_res)[chosen_cs])
            res = np.asarray(rembo_res)
            res = np.reshape(res,(-1,res.shape[-1]))

        num_runs = res.shape[0]
        res_mean = np.mean(res,axis=0)
        res_std = np.std(res,axis=0)

        loss_mat[i_l][b_i][0] = res_mean[-1]

        it = range(res_mean.shape[0])
        t_limits = sp.t.interval(0.95,num_runs) / np.sqrt(num_runs)

        axs[0][i_l].plot(it,res_mean, color+symbol+'-', linewidth=2, label=label)
        axs[0][i_l].fill(np.concatenate([it,it[::-1]]),
                np.concatenate([res_mean + t_limits[0] * res_std,
                                (res_mean + t_limits[1] * res_std)[::-1]]),
                alpha=.3, fc=color, ec='None')
        
        res = np.asarray(asl_curves)
        if n_emb > 1:
            res = res[:,:b]
            res = np.reshape(res,(-1,rembo_n,b))
            rembo_res = []
            for r_i in range(res.shape[0]):
                r = res[r_i]
                cs = list(combinations(r,n_emb))
                cs_res = []
                for c in cs:
                    interleaved_c = np.empty(n_emb*b)
                    for i in range(n_emb):
                        interleaved_c[i::n_emb] = c[i]
                    for i in range(len(interleaved_c)-1):
                        if interleaved_c[i+1]>interleaved_c[i]:
                            interleaved_c[i+1] = interleaved_c[i]
                    cs_res.append(interleaved_c)
                chosen_cs = np.random.choice(len(cs),rembo_n,replace=False)
                rembo_res.append(np.asarray(cs_res)[chosen_cs])
            res = np.asarray(rembo_res)
            res = np.reshape(res,(-1,res.shape[-1]))

        num_runs = res.shape[0]
        res_mean = np.mean(res,axis=0)
        res_std = np.std(res,axis=0)

        loss_mat[i_l][b_i][1] = res_mean[-1]

        axs[1][i_l].plot(it,res_mean, color+symbol+'-', linewidth=2, label=label)
        axs[1][i_l].fill(np.concatenate([it,it[::-1]]),
                np.concatenate([res_mean + t_limits[0] * res_std,
                                (res_mean + t_limits[1] * res_std)[::-1]]),
                alpha=.3, fc=color, ec='None')

        plt.show(block=False)

print(loss_mat)

handles, labels = axs[0][0].get_legend_handles_labels()
fig.legend(handles,labels)
plt.show(block=True)