%% Function to feature mapping funciton to a polynomial features
%% MAPFEATURES(X1, X2) maps the two input features
%% to quadratic features used in the regularization exercise.

%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size

function output = MapFeatures(X1, X2)
    
    % initialize environment
    degree = 6;
    output = ones(size(X1(:, 1)));
    
    for i = 1:degree,
        for j = 0:i,
            
            output(:, end+1) = (X1 .^ (i-j)) .* (X2 .^ j);
        endfor
    endfor
    
endfunction
