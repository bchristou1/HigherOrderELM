function [output] = linearDendrite(varargin)
% Takes as input an input matrix and a weight matrix

if (size(varargin, 2) < 2 || size(varargin, 2) > 2)
    if size(varargin, 2) < 2
        cprintf('red','Error using ');
        cprintf('_red','Linear Dendrite\n');
        cprintf('red','Too few input arguments\n\n');
    else
        cprintf('red','Error using ');
        cprintf('_red','Linear Dendrite\n');
        cprintf('red','Too many input arguments\n\n');
    end
else
    output = varargin{1} .* varargin{2};
end
end

