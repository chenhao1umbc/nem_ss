# this file the python version of rank1 model 

#%% loading dependency
import os
import h5py 
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.signal import stft 

import torch
from torch import nn
import torch.nn.functional as Func
import torch.utils.data as Data
# from unet.unet_model import UNet

# from torch.utils.tensorboard import SummaryWriter
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)

"make the result reproducible"
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print('done loading')

#%% loade data
d = sio.loadmat('v2.mat')
[N,F,J] = d['v'].shape
M = 5          # no of channels
pwr = torch.ones(1, 3)  # signal powers
max_iter = 400
nvar = 1e-6    # noise variance


# %% EM  algorithm for one sample
# reproduce the Matlab result
d = sio.loadmat('x1M3.mat')
x, c = d['x'], d['c']


#%% Neural EM algorithm
data = h5py.File('x5000M5.mat', 'r')
x = torch.tensor(data['x'], dtype=torch.float) # [sample, N, F, channel]
xtr, xcv, xte = x[:4000], x[4000:4500], x[4500:]

