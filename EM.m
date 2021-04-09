function [vj, cjh, Rj] = EM(x, v, opts)
% This function is the main body of the algorithm
% x is shape of [n_c, 1, NF]
% v is shape of [1, 1, NF]
% model is neural network to be trained
% opts contains all the parameters
% original complex likelihood function is
% Product_(n,f) 1/(det(pi*Rx) * e^ -(x-mu)' * inv(Rx) * (x-mu)
%
% here we the real version
% Product_(n,f) (det(2*pi*Rx)^-0.5 * e^ -0.5*(x-mu)' * inv(Rx) * (x-mu)

n_c = opts.n_c;
NF = opts.NF;
J = opts.J;
NFJ = NF * J;
eps = opts.eps;

% init vj
vj = zeros(1, 1, NF, J);
for j = 1:J
    vj(:, :, :, j) = v;
end
vj = reshape(vj, [NFJ,1]);

% init Rj
Rj = zeros(n_c, n_c, NFJ);
for nfj = 1:NFJ
    Rj(:,:,nfj) = eye(n_c);
end

% init Rcj
Rcj = zeros(n_c, n_c, NFJ);
for nfj = 1: NFJ
    Rcj(:,:, nfj) = (vj(nfj)+eps) * Rj(:, :, nfj);
end
Rcj = reshape(Rcj, [n_c, n_c, NF, J]);

% init Rx, shape of [n_c, n_c, NF]
Rx = sum(Rcj, 4);
Rx = (Rx + permute(Rx, [2,1,3]))/2;  % make symetric

%init cjh, I
cjh = zeros(n_c, 1, NF, J);
Rcjh = Rcj;
I = eye(n_c,n_c);
vj = reshape(vj, NF, J);

for epoch = 1:opts.iter
    for i = 1:opts.n_sample
        %% E-step
        %"Calc. Wiener Filter%" shape of [n_c, n_c, NF, J]
        logp = zeros(NF, J);
        for j = 1:J
            for nf = 1:NF
                Wj = Rcj(:, :,nf,j) * inv(Rx(:, :,nf));
                cjh_ = Wj * x(:,:,nf);
                cjh(:, :, nf, j) = cjh_;
                Rh = (I - Wj)*Rcj(:, :,nf,j);
                Rcjh_ = cjh_ * cjh_' + Rh;
                Rcjh_ = (Rcjh_ + permute(Rcjh_, [2,1,3]))/2;
                Rcjh(:, :,nf,j) = Rcjh_;

                %"calc. log P(cj|x; theta_hat), using log to avoid inf problem%" 
                % R = (Rcj**-1 + (Rx-Rcj)**-1)**-1 = (I - Wj)Rcj, The det of a Hermitian matrix is real
                logp(nf,j) = -0.5*log(det(2*pi*Rh));% cj=cjh, e^(0), shape of [n_batch, n_s, n_f, n_t,]
            end
        end


        %% M-step
        %update Rj
        Rj = zeros(n_c, n_c, NF, J);
        for j = 1:J
            for nf = 1:NF
                Rj(:, :, NF, J) = Rcjh(:, :, nf, j)/(vj(nf, j)+eps);
            end
            Rj = sum(Rj, 3)/NF;  % shape of [n_c, n_c, J]
        end

        % update vj
        % to get gradiant, using a func is a must
        [loss,gradval, Rx, Rcj] = ...
        dlfeval(@loss_func, x, cjh, model, gammaj, Rj, opts);

    end
    
lgraph = createUnet(nr,nc);
options = trainingOptions('sgdm',...
    'MiniBatchSize',1,...
    'MaxEpochs',5,...   % Could try fewer or more Epochs
    'InitialLearnRate',1e-3,...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',20,...
    'Shuffle','every-epoch'); 

neural_net = trainNetwork(ds,lgraph,options);
end

   
end %end of the file

