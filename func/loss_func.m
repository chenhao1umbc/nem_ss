function [loss, Rx, Rcj] = loss_func(logp, x, cj, vj, Rj)
% original complex likelihood function is
% Product_(n,f) 1/(det(pi*Rx) * e^ -(x-mu)' * inv(Rx) * (x-mu)
%
% here we the real version
% Product_(n,f) (det(2*pi*Rx)^-0.5 * e^ -0.5*(x-mu)' * inv(Rx) * (x-mu)
% cj shape of [n_c, 1, NF, J]
% x shape of [n_c, 1, NF]

Rcj = zeros(n_c, n_c, NFJ);
for nfj = 1: NFJ
    Rcj(:,:, nfj) = (vj(nfj)+eps) * Rj(:, :, nfj);
end
Rcj = reshape(Rcj, [n_c, n_c, NF, J]);

% shape of [n_c, n_c, NF]
Rx = sum(Rcj, 4);
Rx = (Rx + permute(Rx, [2,1,3]))/2;  % make symetric


cj_, Rcj_ = x[:,None] - cj, Rx[:,None] - Rcj
%"calc log P(x|cj)"
e_part = -1*cj_.transpose(-1, -2).conj()@Rcj_.inverse()@cj_  
det_part = - (2*pi*Rcj_);
%"calc log P(cj)"
e_part_2 = -1*cj.transpose(-1, -2).conj()@Rcj.inverse()@cj  
det_part_2 = - (2*pi*Rcj) ;
log_part = e_part*det_part + e_part_2*det_part_2;

p = exp(logp);
p[p==float('inf')] = 1e38 % avoid the inf 
loss = -sum(p.*log_part, 'all')


end % end of the function