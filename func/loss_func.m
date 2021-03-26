function [loss, grad, Rx, Rcj] = loss_func(logp, x, cj, model, gamma, Rj,opts)
% original complex likelihood function is
% Product_(n,f) 1/(det(pi*Rx) * e^ -(x-mu)' * inv(Rx) * (x-mu)
%
% here we the real version
% Product_(n,f) (det(2*pi*Rx)^-0.5 * e^ -0.5*(x-mu)' * inv(Rx) * (x-mu)
%
% cj shape of [n_c, 1, NF, J]
% x shape of [n_c, 1, NF]
% logp shape of [NF, J]
% model is the jth model
% gamma is dlarray to take gradient, shape [N, F, J]

J = opts.J;
NF = opts.NF;
NFJ = opts.NFJ;
eps = opts.eps;

vj = zeros(N, F, J);
if model{1} == 0 % just gradient    
    for j = 1:J
        vj(:, :, j) = gamma{j};
    end
else
    for j = 1:J
        modelj = model{j};
        vj(:, :, j) = modelj(gamma{j});  % shape of [N, F, J]
    end
end
vj = reshape(vj, [NFJ,1]);

Rcj = zeros(n_c, n_c,NF, J);
for j = 1: J
    for nf = 1:NF
        Rcj(:,:, nfj) = (vj(nf, j)+eps) * Rj(:, :, j);
    end
end

% shape of [n_c, n_c, NF]
Rx = sum(Rcj, 4);
Rx = (Rx + permute(Rx, [2,1,3]))/2;  % make symetric

e_part = zeros(NF, J);
det_part = zeros(NF, J);
e_part_2 = zeros(NF, J);
det_part_2 = zeros(NF, J);
for j = 1:J
    for nf = 1:NF
        cj_ = x(:,:,nf) - cj(:,:,nf,j);
        Rcj_ = Rx(:,:,nf) - Rcj(:,:,nf,j);

        %"calc log P(x|cj)"
        e_part(nf,j) = -1*cj_ * inv(Rcj_) * cj_ ;
        det_part(nf,j) = - log(det(2*pi*Rcj_));

        %"calc log P(cj)"
        e_part_2(nf,j) = -1*cj * inv(Rcj) * cj ;
        det_part_2(nf,j) = - log(det(2*pi*Rcj)) ;
    end
end
log_part = e_part.*det_part + e_part_2.*det_part_2;

p = exp(logp);
p(p==inf) = 1e38; % avoid the inf 
loss = -sum(p.*log_part, 'all');

grad = cell(J);
for j = 1:J
    grad{j} = dlgradient(loss, gamma{j});
end
end % end of the function