%% Function to optimize the parameter vector for linear regression
%% Parameters obtained are parameters at the global optimum
%% IMPORTANT: parameters must be updated all at a time.

function theta = GradientDescent(X, y, theta, alpha, iter)
    
    % initialize necessary variables
    m = length(y);  % number of dimensions of vector y
    J_history = zeros(iter, 1); % keep track of the cost value J
    
    for i = 1:iter,
        
        prediction = X * theta; % X is m * 2 and theta is 2 * 1
        
        % X' is 2 * m and prediction - y is m * 1
        delta = (1 / m) * X' * (prediction - y);
        theta = theta - alpha * delta;  % alpha is the learning rate
        
        % Track value of cost after each iteration
        J_history(i) = CalculateCostValue(X, y, theta);
        
    endfor
    
endfunction
