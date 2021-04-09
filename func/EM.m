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
klog2pi_2 = 2.756815;  % 3*log(pi*2)*0.5

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

%%init cjh, I
cjh = zeros(n_c, 1, NF, J);
Rcjh = Rcj;
I = eye(n_c,n_c);
Rj = reshape(Rj,[n_c, n_c, NF, J]);
vj = reshape(vj, NF, J);
log_l = zeros(opts.iter, 1);

%%
for epoch = 1:opts.iter
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
    Rx = sum(Rcj, 4);
    Rx = (Rx + permute(Rx, [2,1,3]))/2;  % make symetric
    log_l(epoch) = log_likelihood(x, Rx); 

    %% M-step
    %update Rj
    for j = 1:J
        for nf = 1:NF
            Rj(:, :, NF, J) = Rcjh(:, :, nf, j)/(vj(nf, j)+eps);
        end
        Rj = sum(Rj, 3)/NF;  % shape of [n_c, n_c, J]
    end
    % update vj shape of [NF, J]
    for j = 1:J
        for nf = 1:NF
            Rj_inv = Rj(:, :, nf, J);
            temp = Rj_inv * Rcjh(:, :, nf, J);
            vj(nf, j) = sum(diag(temp));
        end
    end
    
    % calc loss function(-Q) and 
    for j = 1:J
        for nf = 1:NF
            Rcj(:,:, nf, j) = (vj(nf,j)+eps) * Rj(:, :, nf,j);
            Rcj_inv = inv(Rcj(:, :, nf, J));
            temp = Rcjh(:, :, nf, J)*Rcj_inv;
            p1 = sum(diag(temp));
            
        end
    end
    
    logpz = 0.5*(Rcjh@Rcj.inverse()).diagonal(dim1=-2, dim2=-1).sum(-1) \
        + 0.5*(Rcj.det() + 1e-30).log() + klog2pi_2
end

   
end %end of the file

