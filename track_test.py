# Import zone

from os.path import join
from os import getcwd

import numpy as np
import torch

from musegan import Generator # The model is defined here
from musegan import clip_samples, used_pitch_classes, tonal_distance
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

note_thresholds = [0.60828567, 0.55597573, 0.54794814]
true_dim = 256 # True dimension of the model's input

reps = 50

np.random.seed(42)

# --------------------------------

drum_solo_ratio = 0.0
drum_n_bass_ratio = 0.0
upc_b = 0.0
upcs_b_avg = []
upcs_b_std = []
upc_g = 0.0
upcs_g_avg = []
upcs_g_std = []
td = 0.0
tds_avg = []
tds_std = []

for _ in range(reps):

    x = np.random.randn(5000,true_dim)
    samples = tensor_to_np(clip_samples(gen(np_to_tensor(x)),note_thresholds))

    has_track  = (np.sum(samples,(2,3))>0).astype(int)
    is_drum_solo = has_track[:,0]-has_track[:,1]-has_track[:,2] == 1
    drum_solo_ratio += np.sum(is_drum_solo)/5000
    is_drum_n_bass = has_track[:,0]-has_track[:,1]+has_track[:,2] == 2
    drum_n_bass_ratio += np.sum(is_drum_n_bass)/5000

    upcs, upcs_stds = used_pitch_classes(samples,verbose=False,std=True)
    upc_g += upcs[0]
    upcs_g_avg.append(upcs[0])
    upcs_g_std.append(upcs_stds[0])
    upc_b += upcs[1]
    upcs_b_avg.append(upcs[1])
    upcs_b_std.append(upcs_stds[1])
    tds, tds_stds = tonal_distance(samples,verbose=False,std=True)
    td += tds[0]
    tds_avg.append(tds[0])
    tds_std.append(tds_stds[0])

drum_solo_ratio /= reps
print("Drum solo ratio:",drum_solo_ratio)
drum_n_bass_ratio /= reps
print("Drum & Bass ratio:", drum_n_bass_ratio)
upc_g /= reps
upc_g_var = np.sum(np.array(upcs_g_avg)**2 + np.array(upcs_g_std)**2)/reps - upc_g**2
upc_g_std = np.sqrt(upc_g_var)
print("UPC Guitar:", upc_g,", std:",upc_g_std)
upc_b /= reps
upc_b_var = np.sum(np.array(upcs_b_avg)**2 + np.array(upcs_b_std)**2)/reps - upc_b**2
upc_b_std = np.sqrt(upc_b_var)
print("UPC Bass:", upc_b,", std:",upc_b_std)
td /= reps
td_var = np.sum(np.array(tds_avg)**2 + np.array(tds_std)**2)/reps - td**2
td_std = np.sqrt(td_var)
print("Tonal Distance:", td, ", std:",td_std)

print("--------------------------------")

print("Dataset values:")

data = np.load("musegan_samples_DBG.npy")
upcs, upcs_stds = used_pitch_classes(data,verbose=False,std=True)
upc_g = upcs[0]
upc_g_std = upcs_stds[0]
upc_b = upcs[1]
upc_b_std = upcs_stds[1]
td, td_std = tonal_distance(data,verbose=False,std=True)
td = td[0]
td_std = td_std[0]
print("UPC Guitar:", upc_g,", std:",upc_g_std)
print("UPC Bass:", upc_b,", std:",upc_b_std)
print("Tonal Distance:", td, ", std:",td_std)
print("Number of samples:",len(data))
