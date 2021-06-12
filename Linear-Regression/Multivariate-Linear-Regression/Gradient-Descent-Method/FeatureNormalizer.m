%% This function normalizes all the features from the training set and
%% set them to a comparable range, so that none of the features can dominate
%% over the prediction all by itself.

function [X_normalized, mu, sigma] = FeatureNormalizer(X)

    % initialize variables to set a initial environment
    X_normalized = X;
    mu = zeros(1, size(X, 2));  % variable for mean values of features
    sigma = zeros(1, size(X, 2));   % variable for std deviation of features
    
    % iterate over each column to get mean value and std deviation of each feature
    for i = 1:size(X, 2),
            
        mu(i) = mean(X(:, i));
        sigma(i) = std(X(:, i));
        X_normalized(:, i) = (X(:, i) - mu(i)) / sigma(i);
            
    endfor

end
