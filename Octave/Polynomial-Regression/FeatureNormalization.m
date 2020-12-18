%% This function normalizes the features in X. It returns a normalized version
%% of X where the mean of each feature is 0 and the std deviation is 1.

function [X_n, mu, sigma] = FeatureNormalization(X)
    
    mu = mean(X);
    X_n = bsxfun(@minus, X, mu);
    
    sigma = std(X_n);
    X_n = bsxfun(@rdivide, X_n, sigma);
    
endfunction
