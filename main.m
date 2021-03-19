% This file is the matlab version with real data
#%% the body of the EM structure
[x, v] = load_data(data='val')
opts = load_options(n_s=v.size(2))
model = init_neural_network(opts)

vj, cj, Rj, neural_net = train_NEM(x, v, model, opts)

# %% test data
vj, cj, Rj, neural_net = test_NEM(v, x, neural_net, opts)