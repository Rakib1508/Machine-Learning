%% This function reads the fixed vocabulary list in vocab.txt and
%% returns a cell array of the words in vocabList.

function vocabList = VocabGenerator()

    %% setup necessary variables
    fid = fopen('Vocabs.txt');
    n = 1899;
    vocabList = cell(n, 1);
    
    for i = 1:n
        % Word Index (can ignore since it will be = i)
        fscanf(fid, '%d', 1);
        % Actual Word
        vocabList{i} = fscanf(fid, '%s', 1);
    endfor
    
    fclose(fid);

endfunction
