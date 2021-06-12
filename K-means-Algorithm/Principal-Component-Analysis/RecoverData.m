%% This function recovers an approximation the original data that has been
%% reduced to K dimensions. It returns the approximate reconstruction in X_rec.

function X_rec = RecoverData(Z, U, K)

    % Initialize required variables
    X_rec = zeros(size(Z, 1), size(U, 1));
    U_reduce = U(:, 1:K);
    
    X_rec = Z * U_reduce';

endfunction
