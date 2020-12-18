%% This function generates the train and cross validation set errors needed 
%% to plot a learning curve of traning errors against cross validation errors

function [error_train, error_cv] = LearningCurve(X, y, Xval, yval, lambda)
    
    % initialize useful variables
    m = size(X, 1);
    error_train = zeros(m, 1);
    error_cv = zeros(m, 1);
    
    for i = 1:m,
        
        X_train = X(1:i, :);
        y_train = y(1:i);
        theta = Trainer(X_train, y_train, lambda);
        error_train(i) = CostFunction(X_train, y_train, theta, 0);
        error_cv(i) = CostFunction(Xval, yval, theta, 0);
        
    endfor
    
endfunction
