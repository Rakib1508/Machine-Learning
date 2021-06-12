%% In this function, cost values are computed from example dataset.
%% We also calculate the gradient vector for each of the examples.

function [J grad] = CostFunction(X, y, theta, lambda)
    
    % Initialize necessary variables
    m = length(y);
    J = 0;
    grad = zeros(size(theta));
    
    h = Sigmoid(X * theta);
    J = - (1 / m) * sum(y .* log(h) + (1 - y) .* log(1 - h)) + (lambda / (2 * m)) * sum(theta(2:end) .^ 2);
    
    theta_reg = [0; theta(2:end)];
    grad = (1 / m) * (X' * (h - y) + lambda * theta_reg);
    grad = grad(:);
    
endfunction
