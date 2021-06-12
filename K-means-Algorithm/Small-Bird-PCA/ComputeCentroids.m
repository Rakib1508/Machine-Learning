%% This function iterates and finds the lowest mean to assign closest
%% centroids to each points from the example dataset.

function centroids = ComputeCentroids(X, idx, K)
    
    % initialize required variables
    [m n] = size(X);
    centroids = zeros(K, n);
    
    for k = 1:K,
        idx_k = (idx == k);
        centroids(k, :) = (idx_k' * X) ./ sum(idx_k);
    endfor
    
endfunction
