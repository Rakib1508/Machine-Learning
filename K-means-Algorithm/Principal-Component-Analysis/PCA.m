%% This function computes eigenvectors of the covariance matrix of X
%% returns the eigenvectors U, the eigenvalues (on diagonal) in S

function [U, S] = PCA(X)

    % Initialize required variables
    [m, n] = size(X);
    U = zeros(n);
    S = zeros(n);

    Sigma = (1 / m) * X' * X;
    [U, S, V] = svd(Sigma);

endfunction
