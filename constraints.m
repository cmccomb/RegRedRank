function [g, h] = constraints(C, X, Y, t)
    g = [];

    % Reshape matrix
    r = size(X, 2);
    s = size(Y, 2);
    C = reshape(C, s, r);
    
    % Compute rank (ish)
    sv = svd(C(:,2:end));
    tol = max(size(C(:,2:end)))*eps(max(s));
    h(1) = sum(sv((t+1):end));
    
%     h(2) = rank(C(:,2:end)) - t;
end