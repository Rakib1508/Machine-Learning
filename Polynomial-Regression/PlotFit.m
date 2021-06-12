%% This function plots a learned polynomial regression fit
%% over an existing figure, also works with linear regression.

function PlotFit(min_x, max_x, mu, sigma, theta, p)
    
    % Hold on to the current figure
    hold on;
    
    % We plot a range slightly bigger than the min and max values to get
    % an idea of how the fit will vary outside the range of the data points
    x = (min_x - 15 : 0.05 : max_x + 25)';
    
    % Map X values
    X_poly = PolynomialFeatures(x, p);
    X_poly = bsxfun(@minus, X_poly, mu);
    X_poly = bsxfun(@rdivide, X_poly, sigma);
    
    % Add ones
    X_poly = [ones(size(x, 1), 1) X_poly];
    
    % Plot
    plot(x, X_poly * theta, '--', 'LineWidth', 2)
    
    % Hold off to the current figure
    hold off;
    
endfunction
