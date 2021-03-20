%This file will generate real(not complex) toy data
% Experiment 1
% Suppose there are 3 classes, the vj is given, 6 Chennels with real "steer vector"
% vj shape of 50*50, the component sources have variation of AOA and power 
% all the x(n,f) the stft is real number
% 1. try Duang's method
% 2. try neural EM with 3 networks

% Experiment 2
% Suppose there are 3 classes, AOA and power are given
% there are variation of vj in each class
% try neural EM with 3 networks to see how \gamma_j works
clear
clc
close all

addpath('func')
rng(0)
visualize = 0;

%% load the original data
importfile('/home/chenhao1/Matlab/nem_ss/data/vj1.png');
vj1 = mean(cdata, 3);
vj1(vj1 == 0) = 3;
importfile('/home/chenhao1/Matlab/nem_ss/data/vj2.png');
vj2 = mean(cdata, 3);
importfile('/home/chenhao1/Matlab/nem_ss/data/vj3.png')
vj3 = mean(cdata, 3);
vj{1} = vj1/norm(vj1);
vj{2} = vj2/norm(vj2);
vj{3} = vj3/norm(vj3);
for j = 1:3 % this make shape has >0 value, non-shape to 0
    vj{j} = max(vj{j}, [], 'all') - vj{j};
end

temp = zeros(50, 50, 3);
for j = 1:3
    temp(:,:,j) = vj{j};
    if visualize
        figure;
        imagesc(vj{j})
    end
end

% for each source the vj(PSD) is normalized to 1
vj = temp; % for later calc., vj will be shape of [50, 50, 3]
if visualize
    figure;
    imagesc(sum(vj,3))
end

%% Experiment 1 data
% real world data is complex time series cj(t)
% it has n_channel channels as cj(t).*steer_vec
% vj(n,f) = sum(|STFT(cj(t).*steer_vec)|.^2)/n_channel

% Here to make sure cj(n,f) is real number, is real number, we have
% cj(n,f) = cj_(n,f).*steer_vec
% vj(n,f) = sum(|cj(n,f)|.^2)/n_channel, where cj(n,f) is the real number
n_channel = 3;
J = 3; % how many sources

aoa = 45; % in degrees
steer_vec = get_steer_vec(aoa, n_channel);
cjnf = zeros(50*50, n_channel, J); % [N*F, n_channel, n_sources]
for j = 1:J
    temp = vj(:,:,j);
    cj_nf = (temp(:)./(steer_vec.*steer_vec)).^0.5; %x >=0
    cjnf(:, :, j) = cj_nf.* sign(rand(50*50, n_channel)-0.5).*steer_vec;
end

% check if vj can be calculated from cjnf
for j = 1:3
v = reshape(sum(abs(cjnf(:,:,j)).^2/n_channel, 2), [50, 50]);
fprintf('check the difference between generated vj and origianl one')
difference = sum(abs(v -vj(:, :, j)) , 'all')  % should be 0
end

% adding power diff
max_db = 20;
power_db = rand(1, 3)*max_db; % power diff for each source
for j = 1:3
    cjnf(:,:,j) = 10^(power_db(j)/20) * cjnf(:, :, j);
end

xnf = sum(cjnf, 3); % sum over all the sources, shape of [N*F, n_channel]
if visualize  % to see the mixture data
    for i = 1:n_channel
        figure;
        imagesc(reshape(xnf(:, i), [50, 50]))
        title(['Channel', num2str(i)])
        colorbar
    end
end

% code Generate online data with diff powers and various ang 
load('./data/vj.mat')
n_channel = 3;
J = 3; % how many sources

aoa = rand(n_channel)*180; % in degrees
steer_vec = get_steer_vec(aoa, n_channel);
cjnf = zeros(50*50, n_channel, J); % [N*F, n_channel, n_sources]
for j = 1:J
    temp = vj(:,:,j);
    cj_nf = (temp(:)./(steer_vec.*steer_vec)).^0.5; %x >=0
    cjnf(:, :, j) = cj_nf.* sign(rand(50*50, n_channel)-0.5).*steer_vec;
end

%% Experiment 2 data



