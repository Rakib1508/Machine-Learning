%% This function plots the dataset along with the decision boundary
%% to show the optimized result

function PlotDecisionBoundary(X, y, theta)
    
    % Plot examples from training data set
    DataPlotter(X(:, 2:3), y);
    hold on;
    
    if size(X, 2) >= 3,
        % Two end points for the decision boundary
        plot_x = [min(X(:, 2)) - 2, max(X(:, 2)) + 2];
        
        % Get decision boundary line
        plot_y = (-1 / theta(3)) .* (theta(2) .* plot_x + theta(1));
        
        plot(plot_x, plot_y);
        legend('Admitted', 'Not admitted', 'Decision boundary');
        axis([30, 100, 30, 100]);
    else
        % Define grid range
        u = linspace(-1, 1.5, 50);
        v = linspace(-1, 1.5, 50);
        
        z = zeros(length(u), length(v));
        
        % evaluate z = X*theta over the grid
        for i = 1:length(u),
            for j = 1:length(v),
            
                z(i, j) = MapFeatures(u(i), v(j)) * theta;
            
            endfor
        endfor
        
        % IMPORTANT: Transpose z before calling contour
        z = z';
        
        % Plot z = 0
        % Notice you need to specify the range [0, 0]
        contour(u, v, z, [0, 0], 'LineWidth', 2);
    end
    
    hold off;
    
endfunction
