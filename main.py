#%%
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#%% the body of the EM structure
x, v = load_data(data='toy1')
opts = load_options(n_s=v.shape[0],n_batch=32, EM_iter=50)
models = init_neural_network(opts)  # a dict of model

vj, cj, Rj, neural_net = train_NEM(x, v, models, opts)
# vj, cj, Rj, neural_net = train_NEM_plain(x, v, opts)


# %%
