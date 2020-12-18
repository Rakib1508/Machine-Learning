%% This function generates the train and validation errors needed to
%% plot a validation curve that we can use to select lambda.

function [lambda_vec, error_train, error_cv] = ...
                    ValidationCurve(X, y, Xval, yval)
    
    % Choose a number of lambda to iterate over
    lambda_vec = [0 0.001 0.003 0.01 0.03 0.1 0.3 1 3 10]';
    error_train = zeros(length(lambda_vec), 1);
    error_cv = zeros(length(lambda_vec), 1);
    X_train = X;
    y_train = y;
    
    for i = 1:length(lambda_vec),
        
        theta = Trainer(X_train, y_train, lambda_vec(i));
        error_train(i) = CostFunction(X_train, y_train, theta, 0);
        error_cv(i) = CostFunction(Xval, yval, theta, 0);
        
    endfor
    
endfunction
