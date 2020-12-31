%% This function returns the cost and gradient for the collaborative
%% filtering problem with the given dataset

function [J, grad] = CostFunction(params, Y, R, num_users, num_movies, ...
                                    num_features, lambda)
    
    % unfold U and W matrices from params
    X = reshape(params(1:num_movies * num_features), num_movies, num_features);
    Theta = reshape(params(num_movies * num_features + 1:end), ...
                    num_users, num_features);
    
    % initialize required variables
    J = 0;
    X_grad = zeros(size(X));
    Theta_grad = zeros(size(Theta));
    
    errors = ((X * Theta' - Y) .* R);
    squaredErrors = errors .^ 2;
    
    J = ((1 / 2) * sum(squaredErrors(:))) + ...
        ((lambda / 2) * sum(Theta(:) .^ 2)) + ((lambda / 2) * sum(X(:) .^ 2));
    
    X_grad = errors * Theta .+ (lambda .* X);
    Theta_grad = errors' * X .+ (lambda .* Theta);
    
    grad = [X_grad(:); Theta_grad(:)];
    
endfunction
