function DataPlotter(x, y)
    
    plot(x, y, 'rx', 'MarkerSize', 10); % rx = red x sign, marker size = 10
    xlabel('City population in 10,000s');
    ylabel('Profit in $10,000s');
    
endfunction
