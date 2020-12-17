%% This function takes a value of z as a argument, then compute and return
%% the gradient of the sigmoid of z to the caller.

function g = SigmoidGradient(z)
    
    g = zeros(size(z));
    g = Sigmoid(z) .* (1 - Sigmoid(z));
    
endfunction
