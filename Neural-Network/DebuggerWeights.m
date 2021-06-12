%% This function randomly initializes a matrix with a given size
%% of rows and columns

function matrix = DebuggerWeights(row, col)
    
    matrix = zeros(row, col + 1);
    
    % Initialize W using "sin", this ensures that W is always of the same
    % values and will be useful for debugging
    matrix = reshape(sin(1:numel(matrix)), size(matrix)) / 10;
    
endfunction
