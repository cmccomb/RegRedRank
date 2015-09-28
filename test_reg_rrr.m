%% Make a new day
close all; 
clear; 
clc

%% Make data
N = 200;
X = randn(N, 12);
Y = [X(:,1)+X(:,2), X(:,3)+0.1*X(:,4), X(:,1)];
Y = Y + 0.25*randn(N, 3);

%% Specify rank and penalty for Lasso
t = 3;
L = 0.00:0.005:0.05;
K = 4;

hold on;

for i=1:1:length(L)
    % Test regularized case
    [br, mse_train, mse_test] = reg_rrr(X, Y, t, L(i), K);
    fprintf('Regularization:\n\tMSE_train = %.3f\n\tMSE_test = %.3f\n', mse_train, mse_test);

    plot(L(i), mse_train, 'ro');
    plot(L(i), mse_test, 'go');
    
    if i==1
        plot([0 L(end)], mse_train*[1 1], 'r--');
        plot([0 L(end)], mse_test*[1 1], 'g--');
    end
    pause(0.001);    
end