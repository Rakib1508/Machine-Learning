%% This function takes features and actual results as input and plots those
%% as points along 2D co-ordinate axes showing the graphical representation
%% of any dataset.

function DataPlotter(X, y)
    
    %Create new diagram
    figure;
    hold on;
    
    positive = find(y == 1);    % get positive examples
    negative = find(y == 0);    % get negative examples
    plot(X(positive, 1), X(positive, 2), 'k+', 'LineWidth', 2, 'MarkerSize', 7);
    plot(X(negative, 1), X(negative, 2), 'ko', 'MarkerFaceColor', 'y',...
            'MarkerSize', 7);
    
endfunction
