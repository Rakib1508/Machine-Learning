function p = PredictOneVsAll(X, theta)
    
    % Initialize useful variables
    m = size(X, 1);
    num_labels = size(theta, 1);
    p = zeros(m, 1);
    
    % Add intercept term to X
    X = [ones(m, 1) X];
    
    prediction = Sigmoid(X * theta');
    [predict_max, index_max] = max(prediction, [], 2);
    p = index_max;
    
endfunction
