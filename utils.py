#@title Adding all the packages
from operator import invert, xor
import os

from numpy.core.fromnumeric import transpose
import h5py 
import numpy as np
import scipy.io as sio
from scipy.signal import stft 
from scipy.signal import istft 
import itertools

import torch
from torch import nn
import torch.nn.functional as Func
import torch.utils.data as Data
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
plt.rcParams['figure.dpi'] = 100

from unet.unet_model import UNetHalf
import torch_optimizer as optim

"make the result reproducible"
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

klog2pi_2 = 2.756815  # 3*np.log(np.pi*2)*0.5

#%%
def load_options(n_s=2, n_epochs=25, n_batch=32):
    """[set all the parameters]

    Args:
        n_s (int, optional): [how many soureces in the mixture]. Defaults to 2.
        n_epochs (int, optional): [how many traning epoches]. Defaults to 25.
        n_batch (int, optional): [batch size]. Defaults to 64.

    Returns:
        [dict]: [a dict contains all the parameters]
    """
    opts = {}
    opts['n_epochs'] = n_epochs 
    opts['lr'] = 0.01
    opts['n_batch'] = n_batch
    opts['n_iter'] = 50 # EM iterations
    opts['d_gamma'] = 16 # gamma dimesion 32*32
    opts['n_s'] = n_s  # number of sources
    return opts

def label_gen(n):
    """This function generates labels in the mixture

    Parameters
    ----------
    n : [int]
        how many components in the mixture
    """
    lb_idx = np.array(list(itertools.combinations([0,1,2,3,4,5], n)))
    label_n = np.zeros( (lb_idx.shape[0], 6) )
    for i in range(lb_idx.shape[0]):
        label_n[i, lb_idx[i]] = 1
    return torch.tensor(label_n).to(torch.float)


def mix_data_torch(x, labels):
    """This functin will mix the data according the to labels

        Parameters
        ----------
        x : [tensor of complex]
            [data with shape of [n_class, n_samples, time_length, n_c]]
        labels : [matrix of int]
            [maxtrix of [n_comb, n_classes]]

        Returns
        -------
        [complex pytorch]
            [mixture data with shape of [n_comb, n_samples, time_len, n_c] ]
    """
    n = labels.shape[0]  # how many combinations
    n_class, n_samples, time_length, n_c = x.shape
    output = torch.zeros( (n, n_samples, time_length, n_c), dtype=torch.cfloat)
    for i1 in range(n):
        s = 0
        for i2 in range(6):  # loop through 6 classes
            if labels[i1, i2] == 1:
                s = s + x[i2]
            else:
                pass
        output[i1] = s
    return output, labels.to(torch.float)


def save_mix(x, lb1, lb2, lb3, lb4, lb5, lb6, pre='_'):
    mix_1, label1 = mix_data_torch(x, lb1)  # output is in pytorch tensor
    mix_2, label2 = mix_data_torch(x, lb2)
    mix_3, label3 = mix_data_torch(x, lb3)
    mix_4, label4 = mix_data_torch(x, lb4)
    mix_5, label5 = mix_data_torch(x, lb5)
    mix_6, label6 = mix_data_torch(x, lb6)

    torch.save({'data':mix_1, 'label':label1}, pre+'dict_mix_1.pt')
    torch.save({'data':mix_2, 'label':label2}, pre+'dict_mix_2.pt')
    torch.save({'data':mix_3, 'label':label3}, pre+'dict_mix_3.pt')
    torch.save({'data':mix_4, 'label':label4}, pre+'dict_mix_4.pt')
    torch.save({'data':mix_5, 'label':label5}, pre+'dict_mix_5.pt')
    torch.save({'data':mix_6, 'label':label6}, pre+'dict_mix_6.pt')


def get_label(lb, shape):
    """repeat the labels for the shape of mixture data

        Parameters
        ----------
        lb : [torch.float matrix]
            [matrix of labels]
        shape : [tuple int]
            [data shape]]

        Returns
        -------
        [labels]
            [large matrix]
    """
    n_comb, n_sample = shape
    label = np.repeat(lb, n_sample, axis=0).reshape(n_comb, n_sample, 6 )
    return label


def get_mixdata_label(mix=1, pre='train_'):
    """loading mixture data and prepare labels

        Parameters
        ----------
        mix : int, optional
            [how many components in the mixture], by default 1

        Returns
        -------
        [data, label]
    """
    dicts = torch.load('../data/data_ss/'+pre+'dict_mix_'+str(mix)+'.pt')
    label = get_label(dicts['label'], dicts['data'].shape[:2])
    return dicts['data'], label


def get_Unet_input(x, l, y, which_class=0, tr_va_te='_tr', n_batch=30, shuffle=True):
    class_names = ['ble', 'bt', 'fhss1', 'fhss2', 'wifi1', 'wifi2']
    n_sample, t_len = x.shape[1:]
    x = x.reshape(-1, t_len)
    l = l.reshape(-1, 6)

    ind = l[:, which_class]==1.0  # find the index, which belonged to this class
    ltr = l[ind]  # find labels

    "get the stft with low freq. in the center"
    f_bins = 200
    f, t, Z = stft(x[ind], fs=4e7, nperseg=f_bins, boundary=None)
    xtr = torch.tensor(np.log(abs(np.roll(Z, f_bins//2, axis=1))))

    "get the cleaned source as the ground-truth"
    f, t, Z = stft(y[which_class], fs=4e7, nperseg=f_bins, boundary=None)
    temp = torch.tensor(np.log(abs(np.roll(Z, f_bins//2, axis=1))))
    n_tile = int(xtr.shape[0]/n_sample)
    ytr = torch.tensor(np.tile(temp, (n_tile, 1,1)))

    data = Data.TensorDataset(xtr, ytr, ltr)
    data = Data.DataLoader(data, batch_size=n_batch, shuffle=shuffle)

    torch.save(data, class_names[which_class]+tr_va_te+'.pt') 
    print('saved '+class_names[which_class]+tr_va_te)   


def awgn(x, snr=20):
    """
        This function is adding white guassian noise to the given signal
        :param x: the given signal with shape of [...,, T], could be complex64
        :param snr: a float number
        :return:
    """
    x_norm_2 = (abs(x)**2).sum()
    Esym = x_norm_2/ x.numel()
    SNR = 10 ** (snr / 10.0)
    N0 = (Esym / SNR).item()
    noise = torch.tensor(np.sqrt(N0) * np.random.normal(0, 1, x.shape), device=x.device)
    return x+noise.to(x.dtype)


def st_ft(x):
    """This is customized stft with np.roll and certain sampling freq.

    Parameters
    ----------
    x : [np.complex or torch.complex64]
        [time series, shape of [...,20100]

    Returns
    -------
    [torch.complex]
        [STFT with shift, shape of 200*200]
    """
    _, _, zm = stft(x, fs=4e7, nperseg=200, boundary=None, return_onesided=False)
    output = np.roll(zm, 100, axis=-2).astype(np.complex)
    return torch.tensor(output).to(torch.cfloat)

#%% EM related functions ####################################################################
def calc_likelihood(x, Rx):
    """Calculate the log likelihood function of mixture x
        p(x|Rx) = \Pi_{n,f} 1/det(pi*Rx) e^{-x^H Rx^{-1} x}
    Parameters
    ----------
    x : [torch.complex]
        [shape of [n_f, n_t, n_c, 1] or [n_f, n_t]]
    Rx : [torch.complex]
        [the covariance matrix, shape of [n_f, n_t, n_c, n_c] or [n_f, n_t]]
    """
    "calculated the log likelihood"
    p1 = -0.5*Rx.det().log() - klog2pi_2
    Rx_1 = torch.linalg.inv(Rx)
    p2 = -0.5* x.transpose(-1, -2) @ Rx_1 @x
    P = p1 + p2.squeeze_()  # shape of [n_f, n_t]
    return P.sum()


def em_simple(init_stft, stft_mix, n_iter):
    """This function is exactly as the norber.expectation_maximization() but only
        works for 1 channel, using pytorch

        Parameters
        ----------
        init_stft : [real tensor]
            [shape of [n_source, f, t]]
        stft_mix : [complex tensor]
            [shape of [n_source, 1, f, t]]
        n_iter : [int]
            [how many iterations for EM]

        Returns
        -------
        [complex tensor]
            [shape of [n_source, f, t]]
    """
    # EM from Norbert for only 1 Channel, 1 sample
    n_s, n_f, n_t = init_stft.shape # number of sources, freq. bins, time bins
    n_c = 1 # number of channels
    cjh = init_stft.clone().to(torch.complex64).exp()
    x = torch.tensor(stft_mix).squeeze()
    eps = 1e-28
    # Rj =  (Rcj/(vj+eps)).sum(2)/n_t  # shape of [n_s, n_f]
    Rj =  torch.ones(n_s, n_f).to(torch.complex64)  # shape of [n_s, n_f]
    likelihood = torch.zeros(n_iter).to(torch.complex64)

    for i in range(n_iter):
        vj = cjh.abs()**2  #shape of [n_s, n_f, n_t], mean of all channels
        # Rcj = cjh*cjh.conj()  # shape of [n_s, n_f, n_t]
        # Rj = cjh@cjh.conj().reshape(n_s, n_t, n_f)/vj.sum(2).unsqueeze(-1)
        
        "Compute mixture covariance"
        Rx = (vj * Rj[..., None]).sum(0)  #shape of [n_f, n_t]
        "Calc. Wiener Filter"
        Wj = vj*Rj[..., None] / (Rx+eps) # shape of [n_s, n_f, n_t]
        "get STFT estimation"
        cjh = Wj * x  # shape of [n_s, n_f, n_t]
        likelihood[i] = calc_likelihood(x, Rx)

    return cjh, likelihood


def em_10paper(init_stft, stft_mix, n_iter):
    """This function is implemented using 2010's paper, for 1 channel with pytorch

        Parameters
        ----------
        init_stft : [real tensor]
            [shape of [n_source, f, t]]

        stft_mix : [complex tensor]
            [shape of [n_source, 1, f, t]]
            
        n_iter : [int]
            [how many iterations for EM]

        Returns
        -------
        [complex tensor]
            [shape of [n_source, f, t]]
    """

    # EM from Norbert for only 1 Channel, 1 sample
    n_s, n_f, n_t = init_stft.shape # number of sources, freq. bins, time bins
    n_c = 1 # number of channels
    cjh = init_stft.clone().to(torch.complex64).exp()  #shape of [n_s, n_f, n_t]
    x = torch.tensor(stft_mix).squeeze()
    eps = 1e-28
    "Initialize spatial covariance matrix"
    Rj =  torch.ones(n_s, n_f).to(torch.complex64)  # shape of [n_s, n_f]
    Rcjh = Rj[..., None] * cjh.abs()**2
    likelihood = torch.zeros(n_iter).to(torch.complex64)

    for i in range(n_iter):
        "Get spectrogram- power spectram"
        vj = Rcjh/Rj[..., None]  #shape of [n_s, n_f, n_t], mean of all channels
        # vj = cjh.abs()**2  #shape of [n_s, n_f, n_t], mean of all channels
        "cal spatial covariance matrix"
        Rj = 1/n_t* (Rcjh/(vj+eps)).sum(-1) # shape of [n_s, n_f]
        "Compute mixture covariance"
        Rx = (vj * Rj[..., None]).sum(0)  #shape of [n_f, n_t]
        likelihood[i] = calc_likelihood(x, Rx)

        Rcj = vj * Rj[..., None] # shape of [n_s, n_f, n_t]
        "Calc. Wiener Filter"
        Wj = Rcj / (Rx+eps) # shape of [n_s, n_f, n_t]
        "get STFT estimation, the conditional mean"
        cjh = Wj * x  # shape of [n_s, n_f, n_t]
        "get covariance"
        Rcjh = cjh.abs()**2 + (1 -  Wj) * Rcj # shape of [n_s, n_f, n_t]

    return cjh, likelihood


def em10(init_stft, stft_mix, n_iter):
    """This function is implemented using 2010's paper, for multiple channels with pytorch

        Parameters
        ----------
        init_stft : [real tensor]
            [shape of [n_source, f, t]]

        stft_mix : [complex tensor]
            [shape of [f, t, n_channel]]
            
        n_iter : [int]
            [how many iterations for EM]

        Returns
        -------
        [complex tensor]
            [shape of [n_s, n_f, n_t, n_c]]
    """
    n_s = init_stft.shape[0]
    n_f, n_t, n_c =  stft_mix.shape 
    I =  torch.ones(n_s, n_f, n_t, n_c).diag_embed().to(torch.complex64)
    eps = 1e-20  # no smaller than 1e-22
    x = torch.tensor(stft_mix).unsqueeze(-1)  #shape of [n_s, n_f, n_t, n_c, 1]
    "Initialize spatial covariance matrix"
    Rj =  torch.ones(n_s, n_f, 1, n_c).diag_embed().to(torch.complex64) 
    vj = init_stft.clone().to(torch.complex64).exp()
    cjh = vj.clone().unsqueeze(-1)  # for n_ter == 0
    cjh_list = []
    for i in range(n_c-1):
        cjh = torch.cat((cjh, vj.unsqueeze(-1)), dim=-1)
    cjh_list.append(cjh.squeeze())
    likelihood = torch.zeros(n_iter).to(torch.complex64)

    for i in range(n_iter):
        Rcj = (vj * Rj.permute(3,4,0,1,2)).permute(2,3,4,0,1) # shape as Rcjh
        # if i != 0 : Rcj = Rcjh  # for debugging 
        "Compute mixture covariance"
        Rx = Rcj.sum(0)  #shape of [n_f, n_t, n_c, n_c]
        "Calc. Wiener Filter"
        Wj = Rcj @ torch.tensor(np.linalg.inv(Rx)) # shape of [n_s, n_f, n_t, n_c, n_c]
        "get STFT estimation, the conditional mean"
        cjh = Wj @ x  # shape of [n_s, n_f, n_t, n_c, 1]
        cjh_list.append(cjh.squeeze())
        likelihood[i] = calc_likelihood(torch.tensor(stft_mix), Rx)

        "get covariance"
        Rcjh = cjh@cjh.permute(0,1,2,4,3).conj() + (I -  Wj) @ Rcj 
        "Get spectrogram- power spectram"  #shape of [n_s, n_f, n_t]
        vj = (torch.tensor(np.linalg.inv(Rj))\
             @ Rcjh).diagonal(dim1=-2, dim2=-1).sum(-1)/n_c
        "cal spatial covariance matrix"
        Rj = ((Rcjh/(vj+eps)[...,None, None]).sum(2)/n_t).unsqueeze(2)

    return cjh_list, likelihood



#%% This section is for the plot functions ##############################################
def plot_x(x, title='Input mixture'):
    """plot log_|stft| of x

    Parameters
    ----------
    x : [np.complex or torch.complex64]
        [time series, shape of 20100]
    """
    if x.shape[-1] == 200:
        y = x
    else:
        y = st_ft(x)

    plt.figure()
    plt.imshow(np.log(abs(y)+1e-30), vmax=-3, vmin=-11,\
         aspect='auto', interpolation='None')
    plt.title(title)
    plt.colorbar()


def plot_log_stft(stft_mix, title="STFT"):
    """plot stft, if stft is out put

    Parameters
    ----------
    x : [np.complex or torch.float32]
        [shape of 200*200]
    """
    plt.figure()
    plt.imshow(stft_mix, vmax=-3, vmin=-11, aspect='auto', interpolation='None')
    plt.title(title)
    plt.colorbar()



#%% this is a new section ##############################################
def load_data(data='toy1'):
    """load data, for train_val data and test data

    Args:
        data (str, optional): [experient 1 or experiment 2]. Defaults to 'toy1'.

    Returns:
        [x, v]: [data and psd]
    """
    if data == 'toy1':
        x = torch.load('./data/x_toy1.pt')  # 20k samples, shape of [i, f, t, channel]
        v = (x.abs()**2).sum(-1).unsqueeze(1)
        v = torch.cat(3*[v], 1)
        return x, v

    elif data == 'toy2':
        #TODO
        pass

    else:
        print('no data found')
        return None


def init_neural_network(opts):
    model = UNetHalf(n_channels=opts['n_s'], n_classes=opts['n_s'])
    if torch.cuda.is_available(): model = model.cuda()
    return model


def train_NEM(X, V, model, opts):
    """This function is the main body of the training algorithm of NeuralEM for Source Separation

    Args:
        i is sample index, total of n_i 
        j is source index, total of n_s
        f is frequecy index, total of n_f
        n is frame(time) index, total of n_t
        m is channel index, total of n_c

        V ([real tensor]): [the initial PSD of each mixture sample, shape of [n_i, n_s, n_f, n_t]]
        X ([complex tensor]): [training mixture samples, shape of [n_i, n_f, n_t, n_c]]
        model ([neural network]): [neural network with random initials]
        opts ([dictionary]): [parameters are contained]

    Returns:
        vj is the updated V, shape of [n_i, n_s, n_f, n_t]
        cj is the source estimation [n_i, n_s, n_f, n_t, n_c]
        Rj is the covariace matrix [n_i, n_s, n_c]
        model is updated neural network

    """
    n_s, n_batch  = V.shape[1], opts['n_batch']
    n_i, n_f, n_t, n_c =  X.shape 
    I =  torch.ones(n_batch, n_s, n_f, n_t, n_c).diag_embed()
    eps = 1e-30  # no smaller than 1e-45
    tr = wrap(X, V, opts)  # tr is a data loader

    optimizer = optim.RAdam(
                    model.parameters(),
                    lr= opts['lr'],
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0)
    loss_train = []
    loss_cv = []

    for epoch in range(opts['n_epochs']):    
        for i, (x, v) in enumerate(tr): # x has shape of [n_batch, n_f, n_t, n_c, 1]
            "Initialize spatial covariance matrix"
            Rj =  torch.ones(n_batch, n_s, 1, 1, n_c).diag_embed()
            "vj is PSD, real tensor, for complex 64 for calc. purpose"
            vj = v#  shape of [n_batch, n_s, n_f, n_t]
            Rcj = (vj * Rj.permute(4,5,0,1,2,3)).permute(2,3,4,5,0,1) # shape as Rcjh
            "Compute mixture covariance"
            Rx = Rcj.sum(1)  #shape of [n_batch, n_f, n_t, n_c, n_c]
            Rx = (Rx + Rx.transpose(-1, -2))/2  # make sure it is symetrix

            if torch.cuda.is_available(): 
                gammaj = torch.ones(n_batch, n_s, opts['d_gamma'],\
                     opts['d_gamma']).cuda().requires_grad_()
            else:
                gammaj = torch.ones(n_batch, n_s, opts['d_gamma'],\
                     opts['d_gamma']).requires_grad_()
            likelihood = torch.zeros(opts['n_iter'])
            optim_gamma = optim.RAdam(
                    [gammaj], # must be iterable
                    lr= opts['lr'],
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0)
            for param in model.parameters():
                param.requires_grad = False

            for ii in range(opts['n_iter']):  # EM loop
                # the E-step
                "for computational efficiency, the following steps are merged in loss function"
                # Rcj = (vj * Rj.permute(4,5,0,1,2,3)).permute(2,3,4,5,0,1) # shape as Rcjh
                # "Compute mixture covariance"
                # Rx = Rcj.sum(1)  #shape of [n_batch, n_f, n_t, n_c, n_c]
                # Rx = (Rx + Rx.transpose(-1, -2).conj())/2  # make sure it is symetrix

                "Calc. Wiener Filter"
                Wj = Rcj @ torch.linalg.inv(Rx)[:,None] # shape of [n_batch, n_s, n_f, n_t, n_c, n_c]
                "get STFT estimation, the conditional mean"
                cjh = Wj @ x[:,None]  # shape of [n_batch, n_s, n_f, n_t, n_c, 1]
                "get covariance"
                Rh = (I - Wj)@Rcj # as Rcjh, shape of [n_batch, n_s, n_f, n_t, n_c, n_c]
                Rcjh = cjh@cjh.permute(0,1,2,3,5,4) + Rh
                Rcjh = (Rcjh + Rcjh.transpose(-1, -2))/2  # make sure it is hermitian (symetrix conj)
                "calc. log P(cj|x; theta_hat), using log to avoid inf problem" 
                # R = (Rcj**-1 + (Rx-Rcj)**-1)**-1 = (I - Wj)Rcj, The det of a Hermitian matrix is real
                logp = -torch.linalg.det(np.pi*Rh).real.log() # cj=cjh, e^(0), shape of [n_batch, n_s, n_f, n_t,]

                # check likihood convergence 
                likelihood[i] = calc_likelihood(x, Rx)

                # the M-step
                "cal spatial covariance matrix" # Rj shape of [n_batch, n_s, 1, 1, n_c, n_c]                
                Rj = ((Rcjh/(vj+eps)[...,None, None]).sum((2,3))/n_t/n_f)[:,:,None,None]
                "Back propagate to update the input of neural network"
                vj = model(gammaj) #shape of [n_batch, n_s, n_f, n_t ]
                loss, Rx, Rcj = loss_func(logp, x, cjh, vj, Rj) # model param is fixed
                optim_gamma.zero_grad()                
                loss.back()
                optim_gamma.step()
                torch.cuda.empty_cache()        

            #%% the neural network step
            gammaj.requires_grad_(False)
            for param in model.parameters():
                param.requires_grad = True
            model.train()
            vj = model(gammaj)           
            loss = loss_func(logp, x, cjh, vj, Rj) # gamma is fixed          
            optimizer.zero_grad()   
            loss.backward()
            optimizer.step()

            loss_train.append(loss.data.item())
            torch.cuda.empty_cache()
            if i%50 == 0: print(f'Current iter is {i} in epoch {epoch}')

        if epoch%1 ==0:
            plt.figure()
            plt.plot(loss_train[-1400::50], '-x')
            plt.title('train loss per 50 iter in last 1400 iterations')

            plt.figure()
            plt.plot(loss_cv, '--xr')
            plt.title('val loss per epoch')
            plt.show()
        
        torch.save(model.state_dict(), './f1_unet'+str(epoch)+'.pt')
        print('current epoch is ', epoch)

        #%% Check convergence
        "if loss_cv consecutively going up for 5 epochs --> stop"
        if check_stop(loss_cv):
            break
    return cjh, vj, Rj, model



def train_NEM_plain(X, V, opts):
    """This function is the main body of the training algorithm of NeuralEM for Source Separation
    only using gradient descent not using nerual networks

    Args:
        i is sample index, total of n_i 
        j is source index, total of n_s
        f is frequecy index, total of n_f
        n is frame(time) index, total of n_t
        m is channel index, total of n_c

        V ([real tensor]): [the initial PSD of each mixture sample, shape of [n_i, n_s, n_f, n_t]]
        X ([real tensor]): [training mixture samples, shape of [n_i, n_f, n_t, n_c]]
        opts ([dictionary]): [parameters are contained]

    Returns:
        vj is the updated V, shape of [n_i, n_s, n_f, n_t]
        cj is the source estimation [n_i, n_s, n_f, n_t, n_c]
        Rj is the covariace matrix [n_i, n_s, n_c]
        model is updated neural network

    """
    n_s  = V.shape[1]
    n_i, n_f, n_t, n_c =  X.shape 
    I =  torch.ones(opts['n_batch'], n_s, n_f, n_t, n_c).diag_embed()
    eps = 1e-20  # no smaller than 1e-45
    tr = wrap(X, V, opts)  # tr is a data loader
    loss_train = []
    likelihood = []
    _, v = next(iter(tr))
    vj = v + eps # v is too clean shape of [n_batch, n_s, n_f, n_t]
    gammaj = (torch.rand(v[0].shape)/10).requires_grad_()
    
    optim_gamma = torch.optim.SGD([gammaj], lr= opts['lr'])
    # optim_gamma = optim.RAdam(
    #         [gammaj], # must be iterable
    #         lr= opts['lr'],
    #         betas=(0.9, 0.999),
    #         eps=1e-8,
    #         weight_decay=0)

    for epoch in range(opts['n_epochs']):    
        for i, (x, _) in enumerate(tr): # x has shape of [n_batch, n_f, n_t, n_c, 1]
            n_batch = x.shape[0]
            "Initialize spatial covariance matrix"
            Rj =  torch.ones(n_batch, n_s, 1, 1, n_c).diag_embed()
            "vj is PSD, real tensor, |xnf|^2" #shape of [n_batch, n_s, n_f, n_t]
            vj = torch.cat(n_batch *[gammaj[None,...]], 0).exp() + eps
            Rcj = (vj.detach() * Rj.permute(4,5,0,1,2,3)).permute(2,3,4,5,0,1) # shape as Rcjh
            "Compute mixture covariance"
            Rx = Rcj.sum(1)  #shape of [n_batch, n_f, n_t, n_c, n_c]
            Rx = (Rx + Rx.transpose(-1, -2))/2  # make sure it is symetrix

            for ii in range(opts['n_iter']):  # EM loop
                # the E-step
                "for computational efficiency, the following steps are merged in loss function"
                # Rcj = (vj * Rj.permute(4,5,0,1,2,3)).permute(2,3,4,5,0,1) # shape as Rcjh
                # "Compute mixture covariance"
                # Rx = Rcj.sum(1)  #shape of [n_batch, n_f, n_t, n_c, n_c]
                # Rx = (Rx + Rx.transpose(-1, -2).conj())/2  # make sure it is symetrix

                "Calc. Wiener Filter"
                Wj = Rcj @ torch.linalg.inv(Rx)[:,None] # shape of [n_batch, n_s, n_f, n_t, n_c, n_c]
                "get STFT estimation, the conditional mean"
                cjh = Wj @ x[:,None]  # shape of [n_batch, n_s, n_f, n_t, n_c, 1]
                "get covariance"
                Rh = (I - Wj)@Rcj # as Rcjh, shape of [n_batch, n_s, n_f, n_t, n_c, n_c]
                Rcjh = cjh@cjh.permute(0,1,2,3,5,4) + Rh
                Rcjh = (Rcjh + Rcjh.transpose(-1, -2))/2  # make sure it is hermitian (symetrix conj)
                "calc. log P(cj|x; theta_hat), using log to avoid inf problem" 
                # R = (Rcj**-1 + (Rx-Rcj)**-1)**-1 = (I - Wj)Rcj, The det of a Hermitian matrix is real
                logp = -0.5*Rh.det().log() - klog2pi_2 # cj=cjh, e^(0), shape of [n_batch, n_s, n_f, n_t,]

                # check likihood convergence 
                likelihood.append(calc_likelihood(x, Rx).item())

                # the M-step
                "cal spatial covariance matrix" # Rj shape of [n_batch, n_s, 1, 1, n_c, n_c]                
                Rj = ((Rcjh/(vj.detach()+eps)[...,None, None]).sum((2,3))/n_t/n_f)[:,:,None,None]
                "update vj"
                # vj = torch.cat(n_batch *[gammaj[None,...]], 0).exp() + eps
                vj = (Rj.inverse() @ Rcjh).diagonal(dim1=-2, dim2=-1).sum(-1)/n_c
                "Back propagate to update the input of neural network"               
                loss, Rx, Rcj = loss_func(logp, x, cjh, vj, Rj) # model param is fixed     
                optim_gamma.zero_grad()    # the neural network/ here only gamma step             
                # loss.backward()
                # print('\nmax gammaj grad before clip', gammaj.grad.abs().max().data)
                # torch.nn.utils.clip_grad_norm_([gammaj], max_norm=500)
                optim_gamma.step()    
                loss_train.append(loss.data.item())
                torch.cuda.empty_cache()
            if i%50 == 0: 
                print(f'Current iter is {i} in epoch {epoch}')
                # print('max gamma, min gamma, max vj, max |gamma.grad|' ,\
                    # gammaj.max().data, gammaj.min().data, vj.max().data, gammaj.grad.abs().max())

        if epoch%1 ==0:
            plt.figure()
            plt.plot(loss_train[-1400::50], '-x')
            plt.title('train loss per 50 iter in last 1400 iterations')
        
        print('current epoch is ', epoch)

        #%% Check convergence
        "if loss_cv consecutively going up for 5 epochs --> stop"
        if check_stop(loss_train):
            break
    return cjh, vj, Rj 


def loss_func(logp, x, cj, vj, Rj):
    """[summary]

    Args:
        logp ([real tensor]): [log likelyhood of P(cj|x; theta_old), shape of [n_batch, n_s, n_f, n_t,]]
        x ([real tensor]): [mixture samples, shape of [n_batch, n_f, n_t, n_c, 1]]
        cj ([real tensor]): [each source, shape of [n_batch, n_s, n_f, n_t, n_c, 1]]
        vj ([real tensor]): [required gradient, similar to PSD of the source, shape of [n_batch, n_s, n_f, n_t ]]
        Rj ([real tensor]): [hidden covariance, shape of [n_batch, n_s, 1, 1, n_c, n_c]]
    Return:
        loss = -1 * \sum_i,n,f \sum_j Q(theta, theta_hat)
    """
    "calc log P(cj, x ; theta) = log P(cj ; theta) + log P(x|cj ; theta)"
    eps = torch.eye(3)*1e-20
    I =  torch.ones(x.shape[:-1]).diag_embed()
    if torch.cuda.is_available():
        eps, I = torch.eye(3, device='cuda')*1e-20, I.cuda()
        logp, x, cj,  vj, Rj = logp.cuda(), x.cuda(), cj.cuda(), vj.cuda(), Rj.cuda()

    Rcj = (vj * Rj.permute(4,5,0,1,2,3)).permute(2,3,4,5,0,1) # shape of [n_batch, n_s, n_f, n_t, n_c, n_c]
    Rcj = (Rcj + Rcj.transpose(-1, -2))/2  # make sure it is hermitian (symetrix conj)
    "Compute mixture covariance"
    Rx = Rcj.sum(1)  #shape of [n_batch, n_f, n_t, n_c, n_c]
    Rx = (Rx + Rx.transpose(-1, -2))/2  # make sure it is hermitian (symetrix conj)

    cj_, Rcj_ = x[:,None] - cj, Rx[:,None] - Rcj + eps # small number to avoid low rank
    "calc log P(x|cj)"
    e_part = -0.5*cj_.transpose(-1, -2)@Rcj_.inverse()@cj_ 
    det_part = -0.5*Rcj_.det().log() - klog2pi_2  # shape of [n_batch, n_s, n_f, n_t]
    "calc log P(cj)"
    e_part_2 = -0.5*cj.transpose(-1, -2)@Rcj.inverse()@cj  
    det_part_2 = -0.5*Rcj.det().log() - klog2pi_2  # shape of [n_batch, n_s, n_f, n_t]
    log_part = e_part.squeeze_() + det_part + e_part_2.squeeze_() + det_part_2

    "Another way calc log P(cj|x) and log P(x)"
    "calc log P(cj|x), because cj equals cjh, epart=0"
    Wj = Rcj @ torch.linalg.inv(Rx)[:,None] 
    Rh = (I - Wj)@Rcj # as Rcjh, shape of [n_batch, n_s, n_f, n_t, n_c, n_c]
    p0 = -0.5*Rh.det().log() - klog2pi_2 # shape of [n_batch, n_s, n_f, n_t]
    "calc log P(x)"
    p1 = -0.5*Rx.det().log() - klog2pi_2
    p2 = -0.5* x.transpose(-1, -2) @ Rx.inverse() @x
    P = p1 + p2.squeeze_() + p0 # shape of [n_f, n_t]
    print('the diff between two ways', (P-log_part).abs().sum(), 'without diff', (P).abs().sum())

    p = logp.exp()  #using logp, instead of p, is because p could be very large number showing inf
    p[p==float('inf')] = 1e38  # roughly the max of float32
    loss = -(p*log_part).sum()
    if loss.isnan():
        error_happens
    return loss, Rx.detach().cpu(), Rcj.detach().cpu()


def check_stop(loss):
    "if loss consecutively goes up for 5 epochs --> stop"
    r = [loss[-i]>loss[-i-1] for i in range(5)]
    rr = True
    for i in r:
        rr = rr and i
    return rr


def wrap(x, v, opts):
    """Wrap X and V for training or testing, in a batch manner
    """
    x = x.unsqueeze_(-1)
    data = Data.TensorDataset(x, v)
    data = Data.DataLoader(data, batch_size=opts['n_batch'], shuffle=True)
    return data


def test_NEM(V, X, model, opts):
    # TODO
    """This function is the main body of the training algorithm of NeuralEM for Source Separation

    Args:
        i is sample index, 
        j is source index, 
        f is frequecy index, 
        n is frame(time) index,
        m is channel index

        V ([real tensor]): [the initial PSD of each mixture sample, shape of [i, f, n]]
        X ([complex tensor]): [training mixture samples, shape of [i, j, f, n, m]]
        model ([neural network]): [neural network with random initials]
        opts ([dictionary]): [parameters are contained]

    Returns:
        vj is the updated V, shape of [i, f, n]
        cj is the source estimation [i, j, f, n, m]
        Rj is the covariace matrix [i, j, m, m]
        model is updated neural network

    """
    loss_cv = []
    model.eval()
    with torch.no_grad():
        cv_loss = 0
        for xval, yval, lval in X: 
            cv_cuda = xval.unsqueeze(1).cuda()
            cv_yh = model(cv_cuda).cpu().squeeze()
            cv_loss = cv_loss + Func.mse_loss(cv_yh, yval)
            torch.cuda.empty_cache()
        loss_cv.append(cv_loss/106)  # averaged over all the iterations


def get_steer_vec(aoa, n_channel, J=3):
    """get 'real number' steer vec

    Args:
        aoa ([vector]): [arriving angles for each source]
        n_channel ([int]): [How many channels]
        J ([int]): [How many sources]
    """
    eps = 1e-30
    elementPos = torch.arange(0, n_channel*0.1-0.1, 0.1)
    c = 299792458
    fc = 1e9
    lam = c/fc

    steer_vec = torch.zeros(J ,n_channel) #[n_sources, n_channel)
    for i in range(J):
        sv = (elementPos*torch.sin(aoa[i]/180*np.pi)/lam*2*np.pi*1j).exp()
        vec = sv.real + eps
        steer_vec[i, :] = vec/ vec.norm()
    return steer_vec