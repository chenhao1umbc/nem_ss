#%% load dependency 
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"]="1"
plt.rcParams['figure.dpi'] = 100
torch.set_printoptions(linewidth=160)
torch.set_default_tensor_type('torch.cuda.DoubleTensor')

#%% load data
# data = h5py.File('data/x5000M5.mat', 'r')
# x = torch.tensor(data['x'], dtype=torch.float) # [sample, N, F, channel]
I = 500 # how many samples
M, N, F, J = 5, 150, 150, 3
NF = N*F
opts = {}
opts['batch_size'] = 64
opts['EM_iter'] = 3
opts['lr'] = 0.01
opts['n_epochs'] = 100  
opts['d_gamma'] = 4 # gamma dimesion 16*16 to 200*200
opts['n_ch'] = 1  

x = torch.rand(I, 150, 150, M, dtype=torch.cdouble)
gamma = torch.rand(I, J, opts['d_gamma'], opts['d_gamma'])
xtr, xcv, xte = x[:int(0.8*I)], x[int(0.8*I):int(0.9*I)], x[int(0.9*I):]
gtr, gcv, gte = gamma[:int(0.8*I)], gamma[int(0.8*I):int(0.9*I)], gamma[int(0.9*I):]
data = Data.TensorDataset(gtr, xtr)
tr = Data.DataLoader(data, batch_size=opts['batch_size'], drop_last=True)

#%% neural EM
model, optimizer = {}, {}
for j in range(J):
    model[j] = UNetHalf(opts['n_ch'], 1).cuda()
    optimizer[j] = optim.RAdam(model[j].parameters(),
                    lr= opts['lr'],
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0)
    for param in model[j].parameters():
        param.requires_grad_(False)

for epoch in range(opts['n_epochs']):    
    for i, (gamma, x) in enumerate(tr): # gamma [n_batch, 4, 4]
        #%% EM part
        "initial"
        g = gamma[:,:,None].cuda().requires_grad_()
        optim_gamma = torch.optim.SGD([g], lr= opts['lr']) 

        vhat = torch.randn(opts['batch_size'], N, F, J).abs().to(torch.cdouble).cuda()
        Hhat = torch.randn(opts['batch_size'], M, J).to(torch.cdouble)
        Rb = torch.ones(opts['batch_size'], M).diag_embed().to(torch.cdouble)*100
        Rxxhat = (x[...,None] @ x[..., None, :].conj()).sum((1,2))/NF
        Rj = torch.zeros(opts['batch_size'], J, M, M).to(torch.cdouble)
        ll_traj = []

        for ii in range(opts['EM_iter']):
            "E-step"
            Rs = vhat.diag_embed() # shape of [I, N, F, J, J]
            Rx = Hhat @ Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() + Rb # shape of [I, N, F, J, J]
            W = Rs.permute(1,2,0,3,4) @ Hhat.transpose(-1,-2).conj() @ Rx.inverse()  # shape of [N, F, I, J, M]
            shat = W.permute(2,0,1,3,4) @ x[...,None]
            Rsshatnf = shat @ shat.transpose(-1,-2).conj() + Rs - (W@Hhat@Rs.permute(1,2,0,3,4)).permute(2,0,1,3,4)

            Rsshat = Rsshatnf.sum([1,2])/NF # shape of [I, J, J]
            Rxshat = (x[..., None] @ shat.transpose(-1,-2).conj()).sum((1,2))/NF # shape of [I, M, J]

            "M-step"
            Hhat = Rxshat @ Rsshat.inverse() # shape of [I, M, J]
            Rb = Rxxhat - Hhat@Rxshat.transpose(-1,-2).conj() - \
                Rxshat@Hhat.transpose(-1,-2).conj() + Hhat@Rsshat@Hhat.transpose(-1,-2).conj()
            Rb = Rb.diagonal(dim1=-1, dim2=-2).diag_embed()
            Rb.imag = Rb.imag - Rb.imag

            # vhat = Rsshatnf.diagonal(dim1=-1, dim2=-2)
            # vhat.imag = vhat.imag - vhat.imag
            for j in range(J):
                vhat[..., j] = model[j](g[:,j]).exp().squeeze()
            loss = loss_func(vhat, Rsshatnf)
            optim_gamma.zero_grad()   
            loss.backward()
            torch.nn.utils.clip_grad_norm_([g], max_norm=500)
            optim_gamma.step()
            torch.cuda.empty_cache()
            vhat = vhat.detach()
            
            "compute log-likelyhood"
            for j in range(J):
                Rj[:, j] = Hhat[..., j][..., None] @ Hhat[..., j][..., None].transpose(-1,-2).conj()
            ll_traj.append(calc_ll_cpx2(x, vhat, Rj, Rb).item())
            print(f'finished {ii} em')
            
        #%% update neural network
        g.requires_grad_(False)
        for j in range(J):
            for param in model[j].parameters():
                param.requires_grad_(True)
            vhat[..., j] = model[j](g[:,j]).exp().squeeze()
            optimizer[j].zero_grad() 

        loss = loss_func(vhat, Rsshatnf)
        loss.backward()
        for j in range(J):
            torch.nn.utils.clip_grad_norm_(model[j].parameters(), max_norm=500)
            optimizer[j].step()
            torch.cuda.empty_cache()

# %%