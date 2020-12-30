%% This function computes the projection of  the normalized inputs X
%% into the reduced dimensional space spanned by the first K columns
%% of U. It returns the projected examples in Z.

function Z = ProjectData(X, U, K)

    % Initialize required variables
    Z = zeros(size(X, 1), K);
    U_reduce = U(:, 1:K);
    
    Z = X * U_reduce;
    
endfunction
