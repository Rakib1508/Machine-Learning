%% This function randomly initializes all the necessary hidden layer and
%% output layer parameters for modeling neural network

function weights = RandomInitialization(row, column)
    
    % initialize required variable
    weights = zeros(column, row + 1);
    epsilon = 0.12;
    
    weights = rand(column, row + 1) * 2 * epsilon - epsilon;
    
endfunction
