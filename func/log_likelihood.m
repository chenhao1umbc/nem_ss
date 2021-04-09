function log_l = log_likelihood(x, Rx)
% clc. the log likelihood
% complex data       p(x;0,Rx) = \Pi_{n,f} 1/det(pi*Rx) e^{-x^H Rx^{-1} x}
% here for real data p(x;0,Rx) = \Pi_{n,f} 1/det(2*pi*Rx)**0.5 e^{-0.5*x^T Rx^{-1} x}
% x shape of [n_c, 1, NF], n_c means number of channels
% Rx shape of [n_c, n_c, NF]

eps = 1e-30;
klog2pi_2 = 2.756815;  % 3*np.log(np.pi*2)*0.5
NF = size(x, 3);
P = zeros(NF, 1);

for nf = 1:NF
    p1 = -0.5*log(det(Rx(:,:,nf))+ eps) - klog2pi_2;
    Rx_inv = inv(Rx(:,:,nf));
    p2 = -0.5 *x(:,:,nf)' * Rx_inv * x(:,:,nf);
    P(nf) = p1 + p2;
end

log_l = sum(P, 'all');

end 