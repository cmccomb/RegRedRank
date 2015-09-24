function mse = objective(C, X, Y, W, L)
    r = size(X, 2);
    s = size(Y, 2);
    C = reshape(C, s, r);
    Y_pred =  X*C';
    er = (Y - Y_pred).^2;
    mse = mean(er(:)) + L*sum(abs(C(:)));
end