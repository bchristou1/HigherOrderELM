function [sigmaPiActivation, sigmaPiDendriteMatrix] = sigmaPiNeuron(varargin)
% Takes as input an input matrix and a weight matrix

if (size(varargin, 2) < 2 || size(varargin, 2) > 2)
    if size(varargin, 2) < 2
        cprintf('red','Error using ');
        cprintf('_red','SigmaPi Neuron\n');
        cprintf('red','Too few input arguments\n\n');
    else
        cprintf('red','Error using ');
        cprintf('_red','SigmaPi Neuron\n');
        cprintf('red','Too many input arguments\n\n');
    end
else
    sigmaPiInputs = varargin{1};
    sigmaPiWeights = varargin{2};
    sigmaPiWeightsMax = max(abs(sigmaPiWeights));
    dimension = size(sigmaPiInputs, 2);
    weightsNo = size(sigmaPiWeights, 2);
    sigmaPiDendriteMatrix = zeros(1, weightsNo);
    sigmaPiActivation = 0;
    for i = 0:weightsNo - 1
        calculateProduct = 1;
        fullMu = dec2bin(i, dimension);
        for j = 1:dimension
            if isequal(fullMu(j), '0')
                mu = -1;
            else
                mu = 1;
            end
            calculateProduct = calculateProduct * (1 + mu * sigmaPiInputs(j));
        end
        sigmaPiDendriteMatrix(i + 1) = (sigmaPiWeights(i + 1) * calculateProduct) / (2 ^ dimension * sigmaPiWeightsMax);
        sigmaPiActivation = sigmaPiActivation + sigmaPiWeights(i + 1) * calculateProduct;
    end
    sigmaPiActivation = sigmaPiActivation / (2 ^ dimension * sigmaPiWeightsMax); 
end
end