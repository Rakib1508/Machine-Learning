%% This function computes the gradient using finite differences
%% and gives us a numerical estimate of the gradient.

function numgrad = NumericalGradient(J, theta)
    
    % initialize necessary items
    numgrad = zeros(size(theta));
    perturb = zeros(size(theta));
    e = 1e-4;
    
    for i = 1:numel(theta),
        
        perturb(i) = e;
        loss1 = J(theta - perturb);
        loss2 = J(theta + perturb);
        
        numgrad(i) = (loss2 - loss1) / (2 * e);
        perturb(i) = 0;
        
    endfor
    
endfunction
