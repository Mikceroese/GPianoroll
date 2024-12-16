# Import zone

from os.path import join
from os import getcwd

import numpy as np
import torch

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

note_thresholds = [0.60828567, 0.55597573, 0.54794814]
true_dim = 256 # True dimension of the model's input

drum_solo_ratio = 0.0
reps = 100

for _ in range(reps):
    x = np.random.randn(3000,true_dim)
    samples = tensor_to_np(clip_samples(gen(np_to_tensor(x)),note_thresholds))
    has_track  = (np.sum(samples,(2,3))>0).astype(int)
    is_drum_solo = has_track[:,0]-has_track[:,1]-has_track[:,2] == 1
    drum_solo_ratio += np.sum(is_drum_solo)/1000

drum_solo_ratio /= reps
print(drum_solo_ratio)



