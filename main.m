% This file is the matlab version with real data
clear
clc
close all

addpath('func')
rng(1)
%% code for generate online data with diff powers and various ang 
load('./data/vj.mat')
J = size(vj,3); % how many sources, J =3
max_db = 20;
n_channel = 3;

aoa = (rand(1,J)-0.5)*90; % in degrees
power_db = rand(1, 3)*max_db; % power diff for each source
steer_vec = get_steer_vec(aoa, n_channel, J);
cjnf = zeros(50*50, n_channel, J); % [N*F, n_channel, n_sources]
for j = 1:J
    temp = vj(:,:,j);
    st_sq = steer_vec(j,:).^2;
    cj_nf = (temp(:)./st_sq).^0.5; %x >=0
    cjnf(:, :, j) = cj_nf.* sign(rand(50*50, n_channel)-0.5).*steer_vec(j,:);
end
for j = 1:J
    cjnf(:,:,j) = 10^(power_db(j)/20) * cjnf(:, :, j);
end
xnf = sum(cjnf, 3); % sum over all the sources, shape of [N*F, n_channel]


%% load options
[N, F, NF] = deal(50, 50, 2500);
opts.n_c = n_channel;  % n_channel=3
opts.iter = 200;
opts.J = 3; % how many sources
opts.N = N;
opts.F = F;
opts.NF = NF;
opts.eps = 1e-30;


%% load neural network
% model = init_neural_network(opts);
model = 0;

%% train NEM
x = reshape(xnf', [opts.n_c, 1,NF]);
v = reshape(sum(abs(x).^2/n_channel, 1), [1,1, NF]);
[vj, cj, Rj, neural_net] = train_NEM(x, v, model, opts);

 %% test data
[vj, cj, Rj, neural_net] = test_NEM(v, x, neural_net, opts);