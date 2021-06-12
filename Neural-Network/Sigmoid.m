%% This function takes a value of z as a argument, then compute and return
%% the sigmoid value to the caller.

function g = Sigmoid(z)
    
    g = 1.0 ./ (1.0 + exp(-z));

endfunction
