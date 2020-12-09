%% This function performs gradient descent over the training examples
%% to find the opitmized parameter vector theta by step-by-step decreasing
%% the cost value to the global minima.

function [theta, J_tracker] = GradientDescent(X, y, theta, alpha, iter)

    % initialize variables to set a initial environment
    m = length(y); % number of training examples
    J_tracker = zeros(iter, 1); % to keep track of cost values

    % update each parameter at one go and the find the cost value for it
    for i = 1:iter,
        
        prediction = X * theta; % X is m * n and theta is n * 1
        delta = (1 / m) * X' * (prediction - y);    % delta is n * 1
        
        % update all parameters in the vector in one go
        theta = theta - alpha * delta;
        
        % Find the cost value for current theta parameter,
        % should decrease after each iteration until global minima
        J_tracker(i) = CalculateCostValue(X, y, theta);

    endfor

end
