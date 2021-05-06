% Gaussian mixture separation via EM
% Toy example with complex valued signals.
% EM algorithm for rank-1 model

%% parameters
clear;
load('v.mat');
[N,F,J] = size(v);
M = 3;          % no of channels
pwr = [10 1 1];  % signal powers
theta = [213 58 35]*pi/180;  % signal AOAs  [-60 10 45]
max_iter = 100;
rseed = 1;      % random number gen seed
nvar = 1e-6;    % noise variance

rng(rseed);

%% generate the signal
h = exp(-1i*pi*(0:M-1)'*sin(theta));
v1 = zeros(N,F,J);
s = zeros(N,F,J);
c = zeros(M,N,F,J);
for j = 1:J
    v1(:,:,j) = pwr(j)*v(:,:,j);
end
for j = 1:J
    s(:,:,j) = (randn(N,F)+1i*randn(N,F))/sqrt(2).*sqrt(v1(:,:,j));
    for m = 1:M
        c(m,:,:,j) = h(m,j)*s(:,:,j);
    end
end
x = sum(c,4) + (randn(M,N,F)+1i*randn(M,N,F))/sqrt(2)*sqrt(nvar);   % M x N x F

%% EM algorithm
vhat = abs(randn(N,F,J));
Hhat = (randn(M,J)+1i*randn(M,J))/sqrt(2);
%Hhat = eye(M,J);
Rb = 1e2*eye(M,M);
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
    iter
    
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
                assert(abs(imag(vhat(n,f,j))/real(vhat(n,f,j))) < 1e-5);
                vhat(n,f,j) = real(vhat(n,f,j));
            end
        end
    end
    Hhat = Rxshat/Rsshat;
    Rb = diag(diag( Rxxhat - Hhat*Rxshat' - Rxshat*Hhat' + Hhat*Rsshat*Hhat'));
    assert(norm(imag(Rb(:))) < 1e-5);
    Rb = real(Rb);
    
    % compute log-likelihood
    for j = 1:J
        Rj(:,:,j) = Hhat(:,j)*Hhat(:,j)';
    end
    ll_traj = [ll_traj calc_ll_cpx2(x,vhat,Rj,Rb)];
    if mod(iter,50) == 0
        figure(100);
        plot(ll_traj,'o-');
        pause(.1);
    end
end

%% display results
for j = 1:J
    figure(j);
    imagesc(vhat(:,:,j));
end