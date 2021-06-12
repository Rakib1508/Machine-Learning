%% This  function computes Gaussian Kernel over a given dataset and
%% returns parameter sim

function sim = GaussianKernel(X1, X2, sigma)
    
    % Setup required vectors
    X1 = X1(:);
    X2 = X2(:);
    sim = 0;
    
    magnitude = sum((X1 - X2) .^ 2);
    sim = e ^ (-magnitude / (2 * sigma ^ 2));
    
endfunction
