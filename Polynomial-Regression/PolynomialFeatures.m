%% This function maps X (1D vector) into the p-th power. It takes data matrix 
%% X of (size m x 1) and maps each example into its polynomial features.

function [X_p] = PolynomialFeatures(X, p)
    
    X_p = zeros(numel(X), p);
    X_p(:, 1) = X;
    
    for i = 2:p,
    
        X_p(:, i) = X .* X_p(:, i - 1);
        
    endfor
    
endfunction
