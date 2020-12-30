%% This function is a helper function that displays the progress of 
%% k-Means as it is running. It is intended for use only with 2D data.
%% It plots the data points with colors assigned to each centroid. 
%% With the previous centroids, it also plots a line between the previous 
%% locations and current locations of the centroids.

function PlotProgresskMeans(X, centroids, previous, idx, K, i)

    % Plot the examples
    PlotDataPoints(X, idx, K);

    % Plot the centroids as black x's
    plot(centroids(:,1), centroids(:,2), 'x', ...
         'MarkerEdgeColor','k', ...
         'MarkerSize', 10, 'LineWidth', 3);

    % Plot the history of the centroids with lines
    for j=1:size(centroids,1)
        DrawLine(centroids(j, :), previous(j, :));
    end

    % Title
    title(sprintf('Iteration number %d', i))

endfunction
