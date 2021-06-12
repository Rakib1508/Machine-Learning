%% This function takes 2D data as argument and plots them on a scaled
%% graph with distinct symbols based on output class.

function DataPlotter(X, y)

    %% Determine marker value
    pos = find(y == 1);
    neg = find(y == 0);

    % Plot Examples
    plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 1, 'MarkerSize', 7);
    hold on;
    plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 7);
    hold off;

endfunction
