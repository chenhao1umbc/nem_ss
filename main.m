% Gaussian mixture separation via EM
% Toy example with real valued signals.
% EM algorithm for rank-1 model

%% parameters
clear;
load('v2.mat');
[N,F,J] = size(v);
M = 3;          % no of channels
pwr = [1 1 1];  % signal powers
max_iter = 400;
rseed = 1;      % random number gen seed
nvar = 1e-6;    % noise variance
rng(rseed);

% use matlab python hybrid coding
myfunc = py.importlib.import_module('myconv');
py.importlib.reload(myfunc);

%% generate the signal
% generate total I of them
I = 5000;
h = randn(M,J, I);
v1 = zeros(N,F,J, I);
s = zeros(N,F,J, I);
c = zeros(M,N,F,J, I);
x = zeros(M, N, F, I);
for j = 1:J
    v1(:,:,j) = pwr(j)*v(:,:,j);
end

for i = 1:I
    for j = 1:J
        s(:,:,j, i) = randn(N,F).*sqrt(v1(:,:,j));
        for m = 1:M
            c(m,:,:,j, i) = h(m,j, i)*s(:,:,j, i);
        end
    end
    x(:,:,:,i) = squeeze(sum(c(:,:,:,:, i),4)) + randn(M,N,F)*sqrt(nvar);   % M x N x F x I
end

%% EM algorithm
% make Neural EM a batch algorithm
vhat = abs(randn(N,F,J)); %vhat = v;
Rb = eye(M,M);
Hhat = randn(M,J);
shat = zeros(J,N,F);
Rsshatnf = zeros(J,J,N,F);
Rxxhat = zeros(M,M);
for n = 1:N
    for f = 1:F
        Rxxhat = Rxxhat + x(:,n,f)*x(:,n,f)';
    end
end
Rxxhat = Rxxhat / N / F;
Rj = zeros(M,M,J);
ll_traj = [];

for iter = 1:max_iter  
    % E-step
    for n = 1:N
        for f = 1:F
            Rsnf = diag(squeeze(vhat(n,f,:)));
            Rxnf = Hhat*Rsnf*Hhat' + Rb;
            Wnf = Rsnf*Hhat'/Rxnf;
            shat(:,n,f) = Wnf*x(:,n,f);
            Rsshatnf(:,:,n,f) = shat(:,n,f)*shat(:,n,f)' + ...
                Rsnf - Wnf*Hhat*Rsnf;
        end
    end
    
    Rsshat = zeros(J,J);
    Rxshat = zeros(M,J);
    for n = 1:N
        for f = 1:F
            Rsshat = Rsshat + Rsshatnf(:,:,n,f);
            Rxshat = Rxshat + x(:,n,f)*shat(:,n,f)';
        end
    end
    Rsshat = Rsshat / N / F;
    Rxshat = Rxshat / N / F;
    
    % M-step
        for j = 1:J
            for n = 1:N
                for f = 1:F
                    vhat(n,f,j) = Rsshatnf(j,j,n,f);
                end
            end
        end
    % use neural network to get vj_hat   
    %vhat= myfunc(gamma);
    Hhat = Rxshat/Rsshat;
    Rb = diag(diag( Rxxhat - Hhat*Rxshat' - Rxshat*Hhat' + Hhat*Rsshat*Hhat'));
    
    % compute log-likelihood
    for j = 1:J
        Rj(:,:,j) = Hhat(:,j)*Hhat(:,j)';
    end
    ll_traj = [ll_traj calc_ll_real2(x,vhat,Rj,Rb)];
    if mod(iter,50) == 0
        figure(100);
        plot(ll_traj,'o-');
        pause(.1);
    end
end

%% display results
for j = 1:J
    figure(j);
    subplot(1,2,1)
    imagesc(vhat(:,:,j));
    colorbar
    
    subplot(1,2,2)
    imagesc(v(:,:,j));
    title('Ground-truth')
    colorbar
end

figure(1001)
imagesc(squeeze(shat(3,:,:)))
caxis([-3, 3])
colorbar

figure(1000)
imagesc(s(:,:,3))
caxis([-3, 3])
colorbar
title('Ground-truth')

norm(squeeze(shat(3,:,:))-s(:,:,3))


