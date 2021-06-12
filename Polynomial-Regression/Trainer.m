%% Trains linear regression given a dataset (X, y) and regularization 
%% parameter lambda trains linear regression using the dataset (X, y)
%%and regularization parameter lambda. Returns the trained parameters theta.

function [theta] = Trainer(X, y, lambda)
    
    % initialize useful variables
    init_theta = zeros(size(X, 2), 1);
    
    % settings for fmincg function call
    CF = @(t) CostFunction(X, y, t, lambda);
    options = optimset('GradObj', 'on', 'MaxIter', 200);
    
    theta = fmincg(CF, init_theta, options);
    
endfunction
