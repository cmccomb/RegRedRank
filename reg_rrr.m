function [beta, mse_train, mse_test] = reg_rrr(X, Y, t, L, K)

    % Constants
    r = size(X, 2)+1;
    s = size(Y, 2);

    % Solve simple rrr for initial guess.
    x0 = rrr(X, Y, 'rank', t);
    x0 = reshape(x0, 1, length(x0(:)));

    X = [ones(size(X, 1), 1) X];

    % Split data for testing, etc
    cv = make_folds(size(X,1), K, size(X,1));

    all_mse_test = ones(1,K);
    all_mse_train = ones(1,K);

    for i=1:1:K
        X_train = X(cv(i).train, :);
        X_test = X(cv(i).test, :);
        Y_train = Y(cv(i).train, :);
        Y_test = Y(cv(i).test, :);

        % Set options
        options = optimoptions('fmincon');
        options.MaxFunEvals = 10000;
        options.Display = 'off';

        % Perform optimization
        x = fmincon(@(x) objective(x, X_train, Y_train, L), x0, [], [], [], [], -Inf*ones(1,39), Inf*ones(1,39), @(x)constraints(x, t), options);

        % Compute the errors
        all_mse_train(i) = objective(x, X_train, Y_train, 0);
        all_mse_test(i) = objective(x, X_test, Y_test, 0);

    end
    
    beta = reshape(x, s, r);
    mse_train = mean(all_mse_train);
    mse_test = mean(all_mse_test);

        function mse = objective(C, X, Y, L)
            C = reshape(C, s, r);
            Y_pred =  X*C';
            er = (Y - Y_pred).^2;
            mse = mean(er(:)) + L*sum(abs(C(:)));
        end


        function [g, h] = constraints(C, t)
            g = [];

            % Reshape matrix
            C = reshape(C, s, r);

            % Compute rank (ish)
            sv = svd(C(:,2:end));
            tol = max(size(C(:,2:end)))*eps(max(s));
            h(1) = sum(sv((t+1):end));
        end
    
    function folds = make_folds(number_of_samples, number_of_folds, max_samples)
        optionss = randsample(max_samples, number_of_samples);
        fold_size = number_of_samples/number_of_folds;
        for ii=1:1:number_of_folds
            temp = optionss;
            test_slice = ((ii-1)*fold_size+1):1:(ii*fold_size);
            folds(ii).test = temp(test_slice);
            temp(test_slice) = [];
            folds(ii).train = temp;
        end
    end
end