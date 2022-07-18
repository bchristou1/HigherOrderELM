function [customHeader] = constructCustomHeader(number, headerName)
customHeader = {};
if size(number, 1) == 1 && size(number, 2) == 1 && floor(number) == number
    customHeader = cell(1, 1);
    customHeader{1} = strcat(headerName, int2str(number));
end

if xor(size(number, 1) > 1, size(number, 2) > 1) && all(all(floor(number) == number))
    customHeader = cell(1, length(number));
    for i = 1:length(number)
        customHeader{i} = strcat(headerName, int2str(number(i)));
    end
end
end

