%% This function performs forward propagation alorithm in order to
%% compute the output of the 2 layered neural network (i.e., 1 hidden layer).

function p = Predict(X, theta1, theta2)
    
    % Initialize necessary variables
    m = size(X, 1);
    num_labels = size(theta2, 1);
    p = zeros(m, 1);
    
    % Apply neural network
    a1 = [ones(m, 1) X];
    
    % layer 2
    z2 = a1 * theta1';
    a2 = [ones(size(z2), 1) Sigmoid(z2)];
    
    % final layer
    z3 = a2 * theta2';
    a3 = Sigmoid(z3);
    
    [predict_max, index_max] = max(a3, [], 2);
    p = index_max;
    
endfunction
