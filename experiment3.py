
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
opts['d_gamma'] = 16 # gamma dimesion 16*16 to 200*200

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
gamma = torch.rand(N, opts['d_gamma'], opts['d_gamma'])
" wrap as data loader"
data = Data.TensorDataset(gamma, data)
tr = Data.DataLoader(data, batch_size=opts['n_batch'], shuffle=True, drop_last=True)

#%% set neural networks
model = UNetHalf(n_channels=1, n_classes=1).cuda()
optimizer = optim.RAdam(model.parameters(),
                lr= opts['lr'],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0)

optim_gamma = torch.optim.SGD([gamma], lr= opts['lr'])

for epoch in range(opts['n_epochs']):    
    for i, (gamma, v) in enumerate(tr): # gamma [n_batch, n_f, n_t]
        out = model(gamma[:,None].cuda())
        loss = ((out - v[:,None].cuda())**2).sum()/opts['n_batch']

        optimizer.zero_grad()   
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=500)
        optimizer.step()
        torch.cuda.empty_cache()
    
    if epoch%1 ==0:
        plt.figure()
        plt.semilogy(loss_cv, '--xr')
        plt.title('val loss per epoch')
        plt.show()

