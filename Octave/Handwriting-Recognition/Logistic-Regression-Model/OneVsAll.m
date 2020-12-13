%% This function optimizes the parameters by running advanced optimzation
%% algorithm 50 iterations over each of the labels.

function [final_theta] = OneVsAll(X, y, num_labels, lambda)
    
    % Initialize useful variables
    m = size(X, 1);
    n = size(X, 2);
    final_theta = zeros(num_labels, n+1);
    
    % Add intercept term to X
    X = [ones(m, 1) X];
    
    intial_theta = zeros(n+1, 1);
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    
    % fmincg implementation:
    for i = 1:num_labels,
        
        % Run fmincg to obtain the optimal theta
        [theta] = fmincg(@(t)(CostFunction(X, (y == i), t, lambda)), intial_theta, options);
        final_theta(i, :) = theta';
        
    endfor
    
endfunction
