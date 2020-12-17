%% Predict the label of an input given a trained neural network outputs
% the predicted label of X given the trained weights of a neural network

function p = Predict(X, theta1, theta2)
    
    % Useful variables
    m = size(X, 1);
    labels = size(theta2, 1);
    p = zeros(size(m, 1));
    
    h1 = Sigmoid([ones(m, 1) X] * theta1');
    h2 = Sigmoid([ones(m, 1) h1] * theta2');
    
    [temp, p] = max(h2, [], 2);
    
endfunction
