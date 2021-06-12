%% In this function, cost values are computed from example dataset.
%% We also calculate the gradient vector for each of the examples.

function [J, grad] = CostFunction(X, y, theta)
    
    % Initialize environment to compute cost and gradient
    m = length(y);
    J = 0;
    grad = zeros(size(theta));
    
    h = Sigmoid(X * theta);
    J = (1 / m) * ((-y' * log(h)) - (1 - y)' * log(1 - h));
    grad = (1 / m) * (X' * (h - y));
    
endfunction
