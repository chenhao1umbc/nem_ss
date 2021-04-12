% This file is the matlab version with real data
clear
clc
close all

addpath('func')
rng(0)
%% code for generate online data with diff powers and various ang 
% check prep_data.m for more information
load('./data/vj.mat')
J = size(vj,3); % how many sources, J =3
max_db = 20;
n_channel = 5;
reproduce_pytorch = false;

if reproduce_pytorch
    n_channel = 3; % more code in EM.m, line 27-34
end
%% generate data
aoa = (rand(1,J)-0.5)*90; % in degrees
power_db = rand(1, J)*max_db; % power diff for each source
steer_vec = get_steer_vec(aoa, n_channel, J);
cjnf = zeros(50*50, n_channel, J); % [N*F, n_channel, n_sources]
for j = 1:J
    temp = vj(:,:,j).^0.5;
    cjnf(:, :, j) = temp(:)./steer_vec(j,:);
end
for j = 1:J
    cjnf(:,:,j) = 10^(power_db(j)/20) * cjnf(:, :, j);
end
xnf = sum(cjnf, 3); % sum over all the sources, shape of [N*F, n_channel]

%% load options
[N, F, NF] = deal(50, 50, 2500);
opts.reproduce_pytorch = reproduce_pytorch;
opts.n_c = n_channel;  % n_channel=3
opts.J = 3; % how many sources
opts.N = N;
opts.F = F;
opts.NF = NF;
opts.eps = 1e-30;
opts.iter = 100;


%% load neural network
% model = init_neural_network(opts);

%% train NEM
x = reshape(xnf', [opts.n_c, 1,NF]);
v = reshape(vj, [NF, J]); %ground truth
% [vj, cj, Rj, neural_net] = train_NEM(x, v, model, opts);
[vj, cj, Rj] = EM(x, v, opts);


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

