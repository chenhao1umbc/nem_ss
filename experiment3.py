
# %% toy experiment 3
"""only generate 1-class vj and let it move. Also train only one neural network to see if it could capture
the patter
"""
#%%
from utils import *

# %% load data
from PIL import Image  # install Pillow
d = Image.open('./data/vj3.png')
d = np.array(d)
d = torch.tensor(1-d[...,0]/255).float()

# row 44 has 1, column 47 has 1
F, T = d.shape
N = 10000
data = torch.zeros(N, F, T)
n = 0
for r in range(105):
    for c in range(102):
        if n < N:
            data[n] = torch.roll(d, (r, c), (0, 1))
            n += 1
gamma = torch.rand(N, 16, 16)

#%% load options for training
opts = {}
opts['n_epochs'] = 100  
opts['lr'] = 0.01
opts['n_batch'] = 64  # batch size
opts['d_gamma'] = gamma.shape[-1] # gamma dimesion 16*16 to 200*200
opts['n_s'] = 1  # number of sources