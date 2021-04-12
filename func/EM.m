function [vj, cjh, Rj] = EM(x, v, opts)
% This function is the main body of the algorithm
% x is shape of [n_c, 1, NF]
% v is shape of [1, 1, NF]
% model is neural network to be trained
% opts contains all the parameters
% original complex likelihood function is
% Product_(n,f) 1/(det(pi*Rx) * e^ -(x-mu)' * inv(Rx) * (x-mu)
%
% here we the real version
% Product_(n,f) (det(2*pi*Rx)^-0.5 * e^ -0.5*(x-mu)' * inv(Rx) * (x-mu)

n_c = opts.n_c;
NF = opts.NF;
J = opts.J;
NFJ = NF * J;
eps = opts.eps;
klog2pi_2 = n_c*log(pi*2)*0.5;  % 3*log(pi*2)*0.5

% init vj
% vj = exp(rand(NF, J)/10);
vj = v;
% for j = 1:J
%     vj(:, j) = sum(v, 2)/J;
% end
vj = abs(awgn(vj, 10));

if opts.reproduce_pytorch
    % % reproduce the pytorch result channel=3
    load('toy1.mat')
    x = reshape(x, [opts.n_c, 1,NF]);
    load('vj.mat')
    vj = reshape(vj, [J, NF]);
    vj = vj';
end

% init Rj
Rj = zeros(n_c, n_c, J);
for j = 1:J
    Rj(:,:,j) = eye(n_c);
end

% init Rcj
Rcj = zeros(n_c, n_c, NF, J);
for j = 1:J
    for nf = 1:NF
        Rcj(:,:, nf, j) = (vj(nf,j)+eps) * Rj(:, :, j);
    end
end

% init cjh, I
cjh = zeros(n_c, 1, NF, J);
Rcjh = Rcj;
I = eye(n_c,n_c);
vj = reshape(vj, NF, J);
log_l = zeros(opts.iter, 1);
loss = zeros(opts.iter, 1);


%%
for epoch = 1:opts.iter
    %% E-step
    Rx = sum(Rcj, 4);  % the last dimension is gone, shape of [n_c, n_c, NF]
    Rx = (Rx + permute(Rx, [2,1,3]))/2;  % make symetric
    for j = 1:J
        for nf = 1:NF
            Wj = Rcj(:, :,nf,j) * inv(Rx(:, :,nf));
            cjh_ = Wj * x(:,:,nf);
            cjh(:, :, nf, j) = cjh_;
            Rh = (I - Wj)*Rcj(:, :,nf,j);
            Rcjh_ = cjh_ * cjh_' + Rh;
            Rcjh_ = (Rcjh_ + permute(Rcjh_, [2,1,3]))/2;
            Rcjh(:, :,nf,j) = Rcjh_; %shape of [n_c, n_c, NF, J]
        end
    end
    log_l(epoch) = log_likelihood(x, Rx); 

    %% M-step
    %update Rj
    Rj = zeros(n_c, n_c, NF, J);
    for j = 1:J
        for nf = 1:NF
            Rj(:, :, nf, j) = Rcjh(:, :, nf, j)/(vj(nf, j)+eps);
        end
    end
    Rj = squeeze(sum(Rj, 3)/NF);% shape of [n_c, n_c, J]

    % update vj shape of [NF, J]
    for j = 1:J
        for nf = 1:NF
            Rj_inv = inv(Rj(:, :, j));
            temp = Rj_inv * Rcjh(:, :, nf, j);
            vj(nf, j) = sum(diag(temp))/n_c;
        end
    end
    
    % calc loss function(-Q)
    % complex data       p(x;0,Rx) = \Pi_{n,f} 1/det(pi*Rx) e^{-x^H Rx^{-1} x}
    % here for real data p(x;0,Rx) = \Pi_{n,f} 1/det(2*pi*Rx)**0.5 e^{-0.5*x^T Rx^{-1} x}
    l = 0;
    for j = 1:J
        for nf = 1:NF
            Rcj(:,:, nf, j) = (vj(nf,j)+eps) * Rj(:, :, j);
            Rcj(:,:, nf, j) = (Rcj(:,:, nf, j)+ Rcj(:,:, nf, j)')/2;
            Rcj_inv = inv(Rcj(:, :, nf, j));
            temp = Rcjh(:, :, nf, j)*Rcj_inv;
            p1 = 0.5*sum(diag(temp));
            p2 = klog2pi_2 + 0.5*log(det(Rcj(:,:,nf,j))+eps);
            l = l + p1 + p2;
        end
    end
    loss(epoch) = l;
    
end

figure;
for j = 1:J
subplot(1,3,j)
imagesc(reshape(vj(:,j), 50, 50));
title(['Source-', num2str(j), ' vj'])
colorbar;
end
   
end %end of the file

