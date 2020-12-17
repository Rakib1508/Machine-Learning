%% This function checks the gradients calculated by back propagation
%% in order to enusre correct implementation of the algorithm.

function CheckGradients(lambda)
    
    if ~exist('lambda', 'var') || isempty(lambda),
        lambda = 0;
    endif
    
    % initialize necessary variables
    input_layer_size = 3;
    hidden_layer_size = 5;
    num_labels = 3;
    m = 5;
    
    % generate random test cases and parameters
    Theta1 = DebuggerWeights(hidden_layer_size, input_layer_size);
    Theta2 = DebuggerWeights(num_labels, hidden_layer_size);
    X = DebuggerWeights(m, input_layer_size - 1);
    y = 1 + mod(1:m, num_labels);
    
    % unroll parameters
    params = [Theta1(:); Theta2(:)];
    
    % Create a shorthand for cost function
    cf = @(p) CostFunction(p, input_layer_size, hidden_layer_size, ...
                           num_labels, X, y, lambda);
    
    [cost, grad] = cf(params);
    numgrad = NumericalGradient(cf, params);
    
    % Visually examine the two gradient computations
    disp([numgrad grad]);
    fprintf(['The above two columns you get should be very similar.\n' ...
             '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);
    
    % Evaluating the norm of the difference between two solutions.  
    % In case of a correct implementation, and assuming EPSILON = 0.0001 
    % in NumericalGradient.m, then diff below should be less than 1e-9
    diff = norm(numgrad - grad) / norm(numgrad + grad);
    
    fprintf(['If your backpropagation implementation is correct, then \n' ...
             'the relative difference will be small (less than 1e-9). \n' ...
             '\nRelative Difference: %g\n'], diff);
    
endfunction
