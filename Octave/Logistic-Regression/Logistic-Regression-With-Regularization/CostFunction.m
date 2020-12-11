%% In this function, cost values are computed from example dataset.
%% We also calculate the gradient vector for each of the examples.

function [J, grad] = CostFunction(X, y, theta, lambda)
    
    % Initialize environment
    m = length(y);
    J = 0;
    grad = zeros(size(theta));
    
    h = Sigmoid(X * theta);
    
    % either of the y and (1 - y) terms are zero.
    J = (1 / m) * sum((-y .* log(h)) - (1 - y) .* log(1 - h)) + ((lambda / (2 * m)) * sum(realpow(theta(2:end), 2)));
    
    theta_reg = theta;
    theta_reg(1) = 0;   % add the intercept term.
    
    % vectorized calculation of gradient with regularization parameter
    grad = ((1 / m) * (X' * (h - y))) + (lambda / m) * theta_reg;
    
endfunction
