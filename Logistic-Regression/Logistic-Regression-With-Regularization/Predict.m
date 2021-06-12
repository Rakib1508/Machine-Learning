%% This function predicts whether the label is 0 or 1 using learned
%% Logistic regression parameter theta

function p = Predict(theta, X)
    
    % set initial values
    m = size(X, 1);
    p = zeros(m, 1);
    
    % If p < 0.5, then p = 0;
    % otherwise p = 1
    p = Sigmoid(X * theta) >= 0.5;
    
endfunction
