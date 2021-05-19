% Gaussian mixture separation via EM
% Toy example with real valued signals.
% EM algorithm for rank-1 model
clear;
rng(1);
load('data/v.mat');
[N,F,J] = size(v);
M = 3;          % no of channels
pwr = [1 1 1];  % signal powers
nvar = 1e-6;    % noise variance
I = 3000; % how many traning samples

%%
vp = zeros(N,F,J);
for j = 1:J
    vp(:,:,j) = pwr(j)*v(:,:,j);
end
% v1 triangel, x:11-34, y 8-32
% v2 circle, x:16-43, y:4-28
% v3 star, x:12-38, y:14-44
n1x = [-1:-1:-10, 1:16]; l_n1x = length(n1x);
n1y = [-1:-1:-7, 1:18]; l_n1y = length(n1y);
n2x = [-1:-1:-15, 1:7]; l_n2x = length(n2x);
n2y = [-1:-1:-3, 1:22]; l_n2y = length(n2y);
n3x = [-1:-1:-11, 1:12]; l_n3x = length(n3x);
n3y = [-1:-1:-13, 1:6]; l_n3y = length(n3y);

x = zeros(I,M,N,F);
for i = 1:I
    % random shift
    vs = vp;   
    temp = randperm(l_n1x);
    vs(:,:,1) = circshift(vp(:,:,1), n1x(temp(1)), 2); % shift y
    temp = randperm(l_n1y);
    vs(:,:,1) = circshift(vs(:,:,1), n1y(temp(1)), 1); % shift x
    
    temp = randperm(l_n2x);
    vs(:,:,2) = circshift(vp(:,:,2), n2x(temp(1)), 2); % shift y
    temp = randperm(l_n2y);
    vs(:,:,2) = circshift(vs(:,:,2), n2y(temp(1)), 1); % shift x
    
    temp = randperm(l_n3x);
    vs(:,:,3) = circshift(vp(:,:,3), n3x(temp(1)), 2); % shift y
    temp = randperm(l_n3y);
    vs(:,:,3) = circshift(vs(:,:,3), n3y(temp(1)), 1); % shift x
    
    theta = (rand(1,M)*300 -150)*pi/180;  % signal AOAs  
    h = exp(-1i*pi*(0:M-1)'*sin(theta));
    s = zeros(N,F,J);
    c = zeros(M,N,F,J);
    for j = 1:J
        s(:,:,j) = (randn(N,F)+1i*randn(N,F))/sqrt(2).*sqrt(vs(:,:,j));
        for m = 1:M
            c(m,:,:,j) = h(m,j)*s(:,:,j);
        end
    end
    x(i,:,:,:) = sum(c,4) + (randn(M,N,F)+1i*randn(M,N,F))/sqrt(2)*sqrt(nvar);   % M x N x F
end
