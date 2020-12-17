%% In this function, cost values are computed from example dataset.
%% We also calculate the gradient vector for each of the examples.
%% Implements the neural network cost function for a two layer
%% neural network which performs classification.

function [J grad] = CostFunction(params, input_layer_size, hidden_layer_size, ...
                          num_labels, X, y, lambda)
    
    % Reshape params back into the parameters, Theta1 and Theta2,
    % the weight matrices for our 2 layer neural network
    Theta1 = reshape(params(1:hidden_layer_size * (input_layer_size + 1)), ...
                     hidden_layer_size, (input_layer_size + 1));
    Theta2 = reshape(params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                     num_labels, (hidden_layer_size + 1));
    
    % Some useful variables
    m = size(X, 1);
    J = 0;
    Theta1_grad = zeros(size(Theta1));
    Theta2_grad = zeros(size(Theta2));
    
    % Implement forward prop to calculate cost values
    % Apply regularization with both
    
    a1 = [ones(m, 1) X];
    
    z2 = a1 * Theta1';
    a2 = [ones(m, 1) Sigmoid(z2)];
    
    z3 = a2 * Theta2';
    a3 = Sigmoid(z3);
    
    h = a3;
    
    Y = zeros(m, num_labels);
    diff = zeros(m, 1);
    
    for i = 1:m,
        
        Y(i, y(i)) = 1;
        diff(i) = log(h(i, :)) * (-Y(i, :))' - log(1 - h(i, :)) * (1 - Y(i, :))';
        
    endfor
    
    J = (1 / m) * sum(diff);
    
    % Add regularization term
    J = J + (lambda / (2 * m)) * (sum(sum(Theta1(:, 2:end) .^ 2)) + sum(sum(Theta2(:, 2:end) .^ 2)));
    
    % Implement backward prop to calculate gradients
    for j = 1:m,
        
        a1 = [1; X(j, :)'];
        
        z2 = Theta1 * a1;
        a2 = [1; Sigmoid(z2)];
        
        z3 = Theta2 * a2;
        a3 = Sigmoid(z3);
        
        grad3 = a3 - Y(j, :)';
        grad2 = (Theta2' * grad3) .* [1; SigmoidGradient(z2)];
        grad2 = grad2(2:end);
        
        Theta1_grad = Theta1_grad + (grad2 * a1');
        Theta2_grad = Theta2_grad + (grad3 * a2');
        
    endfor
    
    Theta1_grad = (1 / m) * Theta1_grad;
    Theta2_grad = (1 / m) * Theta2_grad;
    
    % Apply regularization
    Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + (lambda / m) * Theta1(:, 2:end);
    Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + (lambda / m) * Theta2(:, 2:end);
    
    % Unroll gradients
    grad = [Theta1_grad(:); Theta2_grad(:)];
    
endfunction
