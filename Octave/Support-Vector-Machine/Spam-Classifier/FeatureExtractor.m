%% This function takes in word_indices vector and produces 
%% feature vector x from the word indices

function x = FeatureExtractor(word_indices)
    
    % Necessary variables
    n = 1899;
    x = zeros(n, 1);
    
    % Extract features
    for i = word_indices,
        x(i) = 1;
    endfor

endfunction
