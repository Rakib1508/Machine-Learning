%% This function computes the cost value of the training examples
%% for a given vector of parameters. The cost value is the average of the
%% sum of squared errors between predicted value and actual value for y.

function J = CalculateCostValue(X, y, theta)
    
    % initialize variables to set a initial environment
    m = length(y); % number of training examples
    J = 0;

    % X is m * n and theta is n * 1
    prediction = X * theta;
    squaredErrors = (prediction - y) .^ 2; % squared error for each item
    
    J = (1 / (2 * m)) * sum(squaredErrors);
    
end
