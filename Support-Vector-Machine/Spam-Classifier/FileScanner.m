%% This function reads a file and returns its entire contents in file_contents

function file_contents = FileScanner(filename)

    % Load File
    fid = fopen(filename);
    if fid
        file_contents = fscanf(fid, '%c', inf);
        fclose(fid);
    else
        file_contents = '';
        fprintf('Unable to open %s\n', filename);
    end

endfunction
