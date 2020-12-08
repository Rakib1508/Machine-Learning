%% This function is used to compute the cost value for each example
%% from our training set. The cost value is the squared difference between
%% predicted value and the actual value of y for that example.
%% In a nutshell, J = squared error between prediction and actual value of y

function J = CalculateCostValue(X, y, theta)
    
    % initialize necessary variables
    m = length(y);
    J = 0;
    
    prediction = X * theta; % X is m * 2 and theta is 2 * 1
    squaredError = (prediction - y) .^ 2;
    
    J = (1 / (2 * m)) * sum(squaredError);
    
endfunction
