function [vj, cjh, Rj, neural_net] = train_NEM(x, v, model, opts)
% This function is the main body of the algorithm
% x is shape of [n_c, 1, NF]
% v is shape of [1, 1, NF]
% model is neural network to be trained
% opts contains all the parameters
n_c = opts.n_c;
NF = opts.NF;
J = opts.J;
NFJ = NF * J;
eps = opts.eps;

% init vj
vj = zeros(1, 1, NF, J);
for j = 1:J
    vj(:, :, :, j) = v;
end
vj = reshape(vj, [NFJ,1]);

% init Rj
Rj = zeros(n_c, n_c, NFJ);
for nfj = 1:NFJ
    Rj(:,:,nfj) = eye(n_c);
end

% init Rcj
Rcj = zeros(n_c, n_c, NFJ);
for nfj = 1: NFJ
    Rcj(:,:, nfj) = (vj(nfj)+eps) * Rj(:, :, nfj);
end
Rcj = reshape(Rcj, [n_c, n_c, NF, J]);

% init Rx, shape of [n_c, n_c, NF]
Rx = sum(Rcj, 4);
Rx = (Rx + permute(Rx, [2,1,3]))/2;  % make symetric

%init cjh, I
cjh = zeros(n_c, 1, NF, J);
Rcjh = Rcj
I = eye(n_c,n_c);
vj = reshape(vj, NF, J);

for i = 1:opts.iter
    %% E-step
    %"Calc. Wiener Filter%" shape of [n_c, n_c, NF, J]
    for j = 1:J
        for nf = 1:NF
            Wj = Rcj(:, :,nf,j) * inv(Rx(:, :,nf));
            cjh_ = Wj * x(:,:,nf);
            cjh(:, :, nf, j) = cjh_;
            Rh = (I - Wj)*Rcj(:, :,nf,j);
            Rcjh_ = cjh_ * cjh_' + Rh;
            Rcjh_ = (Rcjh_ + permute(Rcjh_, [2,1,3]))/2;
            Rcjh(:, :,nf,j) = Rcjh_;
        end
    end
    
    %"calc. log P(cj|x; theta_hat), using log to avoid inf problem%" 
    % R = (Rcj**-1 + (Rx-Rcj)**-1)**-1 = (I - Wj)Rcj, The det of a Hermitian matrix is real
    logp = -log(det(pi*Rh))% cj=cjh, e^(0), shape of [n_batch, n_s, n_f, n_t,]

    
    %% M-step
    Rj = zeros(n_c, n_c, NF, J);
    for j = 1:J
        for nf = 1:NF
            Rj(:, :, NF, J) = (Rcjh(:, :, nf, j)/(vj(nf, j)+eps);
            vj = model(gammaj);
        end
    end
    Rj = sum(Rj, 3)/NF;
    
    
%                     % the M-step
%                 %"cal spatial covariance matrix%" % Rj shape of [n_batch, n_s, 1, 1, n_c, n_c]                
%                 Rj = ((Rcjh/(vj+eps)[...,None, None]).sum((2,3))/n_t/n_f)[:,:,None,None]
%                 %"Back propagate to update the input of neural network%"
%                 vj = model(gammaj) %shape of [n_batch, n_s, n_f, n_t ]
%                 loss, Rx, Rcj = loss_func(logp, x, cjh, vj, Rj) % model param is fixed
%                 optim_gamma.zero_grad()                
%                 loss.back()
%                 optim_gamma.step()
%                 torch.cuda.empty_cache()   
    
neural_net = model;
end

   
end

