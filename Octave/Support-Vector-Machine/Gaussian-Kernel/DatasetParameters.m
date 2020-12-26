%% This function computes the optimal values for the parameters required
%% to implement support vector machine training with RBF kernel.

function [C, sigma] = DatasetParameters(X, y, Xval, yval)
    
    % initialize required variables
    C = 1;
    sigma = 0.3;
    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
    min_error = inf;
    
    for C = values,
        for sigma = values,
        
            model = SvmTrainer(X, y, C, @(x1, x2) GaussianKernel(x1, x2, sigma));
            error = mean(double(SvmPredict(model, Xval) ~= yval));
            
            if (error <= min_error),
                C_final = C;
                sigma_final = sigma;
                min_error = error;
                fprintf('new min found C, sigma = %f, %f with error = %f\n\n', ...
                        C_final, sigma_final, min_error);
            endif
        
        endfor
    endfor
    
    C = C_final;
    sigma = sigma_final;
    fprintf(['Best value of C, sigma = [%f, %f]'...
             'with best prediction error = %f\n\n'],C, sigma, min_error);
    
endfunction
