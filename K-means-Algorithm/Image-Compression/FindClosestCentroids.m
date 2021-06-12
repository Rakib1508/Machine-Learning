%% This function takes a dataset X and an initial set of points, with these
%% parameters, it tries to fit all the points of the dataset to their
%% respective closest centroids.

function idx = FindClosestCentroids(X, centroids)
    
    % initialize required variables
    K = size(centroids, 1);
    m = size(X, 1);
    idx = zeros(m, 1);
    
    for i = 1:m,
        for j = 1:K,
            dist(j) = sum((X(i, :) - centroids(j, :)) .^ 2);
        endfor
        [~, idx(i)] = min(dist);
    endfor
    
endfunction
