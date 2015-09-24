close all; clear; clc


% Make data
N = 100;
X = randn(N, 12);
Y = [X(:,1)+X(:,2), X(:,3)+0.1*X(:,4).^2, X(:,1) + 0.5*randn(N,1)];
Y = Y + 0.25*randn(N, 3);

X1 = X(1:(N/2), :);
Y1 = Y(1:(N/2), :);
X2 = X((N/2 + 1):end, :);
Y2 = Y((N/2 + 1):end, :);

t = 3;

% Solve simple rrr for initial guess.
options = optimoptions('fmincon');
% options.Algorithm = 'active-set';
options.MaxFunEvals = 10000;
W = eye(4);
x0 = rrr(X1, Y1, 'rank', t, 'weighting', eye(3));
x0 = reshape(x0, 1, 39);
X1 = [ones(N/2,1) X1];
X2 = [ones(N/2,1) X2];

L = 0.11;

x = fmincon(@(x)objective(x, X1, Y1, W, L), x0, [], [], [], [], -1000*ones(1,39), 1000*ones(1,39), @(x)constraints(x, X1, Y1, t), options);

fprintf('\n%.2f, %.2f\n', objective(x0, X1, Y1, W, 0),    objective(x, X1, Y1, W, 0));
fprintf('\n%.2f, %.2f\n', objective(x0, X2, Y2, W, 0),    objective(x, X2, Y2, W, 0));


hold on;
plot(abs(x), 'ro-');
plot(abs(x0), 'g*-');
