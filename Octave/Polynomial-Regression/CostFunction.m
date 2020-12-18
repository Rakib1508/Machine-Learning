%% This function computes the cost value for training set using the
%% average sum of squared errors between prediction and actual output.
%% In the end, it also calculates the gradient parameters

function [J grad] = CostFunction(X, y, theta, lambda)
    
    % Setup useful variables
    m = length(y);
    J = 0;
    grad = zeros(size(theta));
    
    h = X * theta;
    theta_reg = [0; theta(2:end, :)];
    
    % average squared error
    J = (1 / (2 * m)) * sum((h - y) .^ 2);
    
    % add regularization
    J = J + (lambda / (2 * m)) * theta_reg' * theta_reg;
    grad = (1 / m) * (X' * (h - y) + lambda * theta_reg);
    
endfunction
