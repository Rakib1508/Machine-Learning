%% This function finds the best threshold to use for selecting outliers
%% based on the results from validation set (pval) and the ground truth (yval)

function [bestEpsilon bestF1] = ThresholdFinder(yval, pval)
    
    % initialize required variables
    bestEpsilon = 0;
    bestF1 = 0;
    F1 = 0;
    step = (max(pval) - min(pval)) / 1000;
    
    for epsilon = min(pval):step:max(pval),
        
        prediction = (pval < epsilon);
        tp = sum(prediction == 1 & yval == 1);
        fp = sum(prediction == 1 & yval == 0);
        fn = sum(prediction == 0 & yval == 1);
        
        precision = tp / (tp + fp);
        recall = tp / (tp + fn);
        
        F1 = (2 * precision * recall) / (precision + recall);
        
        if F1 > bestF1,
            bestF1 = F1;
            bestEpsilon = epsilon;
        endif
    
    endfor
    
endfunction
