%% Make a new day
close all; 
clear; 
clc

%% Make data
N = 100;
X = randn(N, 12);
Y = [X(:,1)+X(:,2), X(:,3)+0.1*X(:,4).^2, X(:,1) + 0.5*randn(N,1)];
Y = Y + 0.25*randn(N, 3);

%% Specify rank and penalty for Lasso
t = 3;
L = 0.11;
K = 2;

% Test regularized case
[~, mse_train, mse_test] = reg_rrr(X, Y, t, L, K);
fprintf('Regularization:\n\tMSE_train = %.3f\n\tMSE_test = %.3f\n', mse_train, mse_test);

% Non-regularized case
[~, mse_train, mse_test] = reg_rrr(X, Y, t, 0.0, K);
fprintf('No regularization:\n\tMSE_train = %.3f\n\tMSE_test = %.3f\n', mse_train, mse_test);
