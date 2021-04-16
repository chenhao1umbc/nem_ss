% This file is the matlab version with real data
clear
clc
close all

addpath('func')
rng(0)
%% code for generate online data with diff powers and various ang 
% check prep_data.m for more information
reproduce_pytorch = false;
load('./data/vj.mat')
J = size(vj,3); % how many sources, J =3
eps = 1e-30;
max_db = 0;
M = 3;

if reproduce_pytorch
    M = 3; % more code in EM.m, line 27-34
end
%% generate data
Rj = zeros(M,M,J);
cjnf = zeros(M,1, 50, 50, J); % [n_channel,,F, N, n_sources]
for j = 1:J
%     temp = rand(n_channel);
    temp = rand(M,1);
    Rj(:,:,j) = temp*temp';
    Rj(:,:,j) = Rj(:,:,j)/ norm(Rj(:,:,j), 'fro');
%     rank(Rj(:,:,j))
    
    for f = 1:50
        for n = 1:50
            Rcj = (vj(f,n,j)+eps)*Rj(:,:,j);
            cjnf(:,1,f,n,j) = mvnrnd(zeros(M,1),Rcj);
        end
    end
end
power_db = rand(1, J)*max_db; % power diff for each source
for j = 1:J
    cjnf(:,1,:,:,j) = 10^(power_db(j)/20) * cjnf(:,1 ,:, :, j);
end
xnf = sum(cjnf, 5); % sum over all the sources, shape of [N*F, n_channel]

%% load options
[N, F, NF] = deal(50, 50, 2500);
opts.reproduce_pytorch = reproduce_pytorch;
opts.n_c = M;  % n_channel=3
opts.J = 3; % how many sources
opts.N = N;
opts.F = F;
opts.NF = NF;
opts.eps = eps;
opts.iter = 500;

%% train NEM
x = reshape(xnf, [opts.n_c, 1,NF]);
v = reshape(vj, [NF, J]); %ground truth
[vj, cj, Rjh] = EM(x, v, opts);
% [vj1, cj1, Rjh1] = rank1(x, v, opts);

figure;
for j = 1:J
subplot(1,4,j)
imagesc(reshape(cj(1,1,:, j), 50, 50))
title(['The first channel of Source-', num2str(j), ' cj'])
colorbar;
caxis([0,max(x(1,1,:), [], 'all')])
end
subplot(1,4,4)
imagesc(reshape(x(1,1,:), 50, 50))
title(['The first channel of given mixture xnf'])
colorbar;
caxis([0,max(x(1,1,:), [], 'all')])

%ground truth
figure;
for j = 1:J
subplot(1,4,j)
imagesc(squeeze(cjnf(1,1,:, :,j)))
title(['The first channel of Source-', num2str(j), ' cj'])
colorbar;
caxis([0,max(x(1,1,:), [], 'all')])
end
subplot(1,4,4)
imagesc(reshape(x(1,1,:), 50, 50))
title(['The first channel of given mixture xnf'])
colorbar;
caxis([0,max(x(1,1,:), [], 'all')])
