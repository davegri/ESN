clear all;
    
%% ESN Parameters

% dimensions
n_train_samples = 10000;
n_resevoir = 500;
n_out = 100;
n_in = 1;

% network hyperparameters
alpha = 0.98; % leakage
beta = 1e-10; % ridge regularization parameter
p = 0.99; % spectral radius;
sparsity = 0.4; % percentage of weights in resevoir set to 0
w = 0.0001; % input weight strength
k = 1; % resevoir weight strength


%% create samples and teacher
[X_train, Y0_train] = generate_samples(n_in, n_train_samples, n_out);

%% Generate Weights and Resevoir

% generate weights
W = w*randn(n_resevoir, n_in); % input weights
K = k*randn(n_resevoir,n_resevoir); % resevoir weights
J = randn(n_out, n_resevoir + n_in + 1); % output weights

% make resevoir sparse
zero_weights = ceil(length(K(:))*sparsity);
K(randperm(length(K(:)),zero_weights)) = 0;

% ensure ESP
K = p/max(real(eig(K)))*K;

%% Harvest Resevoir States and generate outputs
Z = harvest_resevoir(X_train, W, K, J, alpha);

%% Compute Output Weights with Ridge Regression
C = Z*Z'; % Input-Input Correlation Matrix
U = Z*Y0_train'; % Input-Output Correlation Matrix
J = transpose((C+beta*eye(size(C)))^(-1)*U); % Update output weights

%% Calculate new outputs after regression

[X_valid, Y0_valid] = generate_samples(n_in, n_train_samples, n_out);
[~, Y_valid] = harvest_resevoir(X_valid, W, K, J, alpha);

%% Evaluate performance of ESN

R2 = calc_R2(Y_valid, Y0_valid);
MC = sum(R2);

% Plot R2
plot(R2);
ylabel("$R^{2}$", "interpreter", "latex");
xlabel("index of output neuron", "interpreter", "latex");
title("$R^{2}$ with respect to index of output neuron, using validation samples", "interpreter", "latex");
subtitle("memory capacity: " + MC, "interpreter", "latex")

function [Z, Y]  = harvest_resevoir(X, W, K, J, alpha)
    % Generate and discard inital resevoir state
    R = zeros(size(W,1), size(X, 2));
    R(:,1) = zeros(size(W,1), 1);
    R(:,1) = (1-alpha)*R(:,1) + alpha*tanh(W*X(:,1)+K*R(:,1));

    Y = zeros(size(J,1),length(X));

    % harvest all resevoir states and output
    for i=1:length(X)
        Z(:,i) = [1;X(:,i);R(:,i)]; % extended resevoir state with bias
        Y(:,i) = J*Z(:,i); % output
        if i<length(X)
            R(:,i+1) = (1-alpha)*R(:,i) + alpha*tanh(W*X(:,i+1)+K*R(:,i));
        end
    end
end

function [X, Y0] = generate_samples(n_in, n_train_samples, n_out)
    % create samples
    all_samples = randn(n_in, n_train_samples);
    X = all_samples(n_out + 1:end);

    % create teacher
    Y0 = zeros(100, length(X));
    for i=1:length(X)-1
        Y0(:,i) = all_samples(n_out+i-1:-1:i)';
    end
end

function [R2] = calc_R2(Y, Y0)
    Y0_mean = mean(Y0,2);
    SS_tot = sum((Y0-Y0_mean).^2,2);
    SS_res = sum((Y-Y0).^2,2);
    R2 = 1-SS_res./SS_tot;
end


