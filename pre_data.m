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
visualize = 0;

%% load the original data
importfile('/home/chenhao1/Matlab/nem_ss/vj1.png');
vj1 = mean(cdata, 3);
vj1(vj1 == 0) = 3;
importfile('/home/chenhao1/Matlab/nem_ss/vj2.png');
vj2 = mean(cdata, 3);
importfile('/home/chenhao1/Matlab/nem_ss/vj3.png')
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

vj = temp; % for later calc., vj will be shape of [50, 50, 3]
if visualize
    figure;
    imagesc(sum(vj,3))
end

%% Experiment 1 data
% vj(n,f) = sum(|x(n,f).*steer_vec|.^2)/n_channel, where x(n,f) is the real number
n_channel = 6;
J = 3; % how many sources

aoa = 45; % in degrees
steer_vec = get_steer_vec(aoa, n_channel);
x = zeros(50*50, n_channel, J);
for j = 1:J
    temp = vj(:,:,j);
    x(:, :, j) = ((temp(:)*n_channel).^0.5)./steer_vec;
end

% check if vj
for j = 1:3
a{j} = abs(x(:,:,j).*steer_vec).^2;
end
v1 = reshape(sum(a{1}, 2)/n_channel, [50, 50]);
sum(abs(v1 -vj(:, :, 1)) , 'all')  % should be 0


%% Experiment 2 data



