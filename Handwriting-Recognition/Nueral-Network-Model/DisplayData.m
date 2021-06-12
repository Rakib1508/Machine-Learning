%% This function takes an example dataset and displays the numeric data
%% as human readable form with grayscale images

function [h, array] = DisplayData(X, width)

    % Set width automatically if not passed in
    if ~exist('width', 'var') || isempty(width) 
        width = round(sqrt(size(X, 2)));
    end

    colormap(gray); % grayscale images

    % Compute rows, columns and height of each item
    [m n] = size(X);
    height = (n / width);

    % Compute number of items to display
    rows = floor(sqrt(m));
    cols = ceil(m / rows);

    % spacing between items
    pad = 1;

    % Setup blank display
    array = - ones(pad + rows * (height + pad), ...
                           pad + cols * (width + pad));

    % Copy each example into a patch on the display array
    current = 1;
    for j = 1:rows,
        for i = 1:cols,
            if current > m, 
                break; 
            endif
            
            % Get the max value of the patch
            max_value = max(abs(X(current, :)));
            array(pad + (j - 1) * (height + pad) + (1:height), ...
                          pad + (i - 1) * (width + pad) + (1:width)) = ...
                            reshape(X(current, :), height, width) / max_value;
            current = current + 1;
        endfor
        
        if current > m, 
            break; 
        endif
    endfor

    h = imagesc(array, [-1 1]); % display example data
    axis image off; % don't show image axis
    drawnow;

endfunction
