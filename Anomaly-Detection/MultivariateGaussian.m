%% This function computes the probability density function of the the examples 
%% X under the multivariate gaussian distribution. The distribution with
%% parameters mu and Sigma2. If Sigma2 is a matrix, it is treated as the
%% covariance matrix. If Sigma2 is a vector, it is treated as the \sigma^2 
%% values of the variances in each dimension (a diagonal covariance matrix)

function p = MultivariateGaussian(X, mu, sigma)
    
    % initialize required variables
    k = length(mu);
    
    if (size(sigma, 2) == 1) || (size(sigma, 1) == 1)
        sigma = diag(sigma);
    endif

    X = bsxfun(@minus, X, mu(:)');
    p = (2 * pi) ^ (- k / 2) * det(sigma) ^ (-0.5) * ...
         exp(-0.5 * sum(bsxfun(@times, X * pinv(sigma), X), 2));

endfunction
