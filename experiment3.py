
# %% toy experiment 3
"""only generate 1-class vj and let it move. Also train only one neural network to see if it could capture
the patter
"""
from utils import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

#%% load options for training 
opts = {}
opts['n_epochs'] = 100  
opts['lr'] = 0.01
opts['n_batch'] = 64  # batch size
opts['d_gamma'] = 2 # gamma dimesion 16*16 to 200*200

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
gamma = torch.rand(N, opts['d_gamma'])
" wrap as data loader"
data = Data.TensorDataset(gamma, data)
tr = Data.DataLoader(data, batch_size=opts['n_batch'], shuffle=True, drop_last=True)

#%% set neural networks
model = UNetHalf2(n_channels=1, n_classes=1).cuda()
optimizer = optim.RAdam(model.parameters(),
                lr= opts['lr'],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0)
for param in model.parameters():
    param.requires_grad_(False)
loss_cv = []

for epoch in range(opts['n_epochs']):    
    for i, (gamma, v) in enumerate(tr): # gamma [n_batch, n_f, n_t]
        x = gamma[:,None].cuda().requires_grad_()
        v = v[:,None].cuda()

        "update gamma"
        optim_gamma = torch.optim.SGD([x], lr= opts['lr'])  # every iter the gamma grad is reset
        out = model(x.diag_embed())
        loss = ((out - v)**2).sum()/opts['n_batch']
        optim_gamma.zero_grad()   
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x], max_norm=500)
        optim_gamma.step()
        torch.cuda.empty_cache()

        "update model"
        for param in model.parameters():
            param.requires_grad_(True)
        x.requires_grad_(False)
        out = model(x.diag_embed())
        loss = ((out - v)**2).sum()/opts['n_batch']
        optimizer.zero_grad()   
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=500)
        optimizer.step()
        torch.cuda.empty_cache()

        loss_cv.append(loss.detach().item())
    
    if epoch%1 ==0:
        plt.figure()
        plt.semilogy(loss_cv, '--xr')
        plt.title('val loss per epoch')
        plt.show()


# %%
