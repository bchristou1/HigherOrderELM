function [multiCubeSigmaPiActivation, multiCubeSigmaPiDendriteMatrix] = multiCubeSigmaPiNeuron(varargin)
% Takes as input an input matrix, a weight matrix and the sub-cubes dimensions

if (size(varargin, 2) < 3 || size(varargin, 2) > 3)
    if size(varargin, 2) < 3
        cprintf('red','Error using ');
        cprintf('_red','Multi-Cube SigmaPi Neuron\n');
        cprintf('red','Too few input arguments\n\n');
    else
        cprintf('red','Error using ');
        cprintf('_red','Multi-Cube SigmaPi Neuron\n');
        cprintf('red','Too many input arguments\n\n');
    end
else
    multiCubeSigmaPiInputs = varargin{1};
    multiCubeSigmaPiWeights = varargin{2};
    multiCubeSigmaPiWeightsMax = max(abs(multiCubeSigmaPiWeights));
    cubesDimensions = varargin{3};
    fullWeightsNo = size(multiCubeSigmaPiWeights, 2);
    multiCubeSigmaPiDendriteMatrix = zeros(1, fullWeightsNo);
    multiCubeSigmaPiActivation = 0;
    count = 0;
    countInputs = 0;
    countWeights = 0;
    for currentCube = 1:size(cubesDimensions, 2)
        subCubeDimension = cubesDimensions(currentCube);
        subCubeWeightsNo = 2 ^ subCubeDimension;
        countInputs = countInputs + subCubeDimension;
        countWeights = countWeights + subCubeWeightsNo;
        subCubeInputs = multiCubeSigmaPiInputs(countInputs - subCubeDimension + 1:countInputs);
        subCubeWeights = multiCubeSigmaPiWeights(countWeights - subCubeWeightsNo + 1:countWeights);
        subCubeActivation = 0;
        for i = 0:subCubeWeightsNo - 1
            count = count + 1;
            calculateProduct = 1;
            fullMu = dec2bin(i, subCubeDimension);
            for j = 1:subCubeDimension
                if isequal(fullMu(j), '0')
                    mu = -1;
                else
                    mu = 1;
                end
                calculateProduct = calculateProduct * (1 + mu * subCubeInputs(j));
            end
            multiCubeSigmaPiDendriteMatrix(count) = (subCubeWeights(i + 1) * calculateProduct) / (2 ^ subCubeDimension * multiCubeSigmaPiWeightsMax);
            subCubeActivation = subCubeActivation + subCubeWeights(i + 1) * calculateProduct;
        end
        multiCubeSigmaPiActivation = multiCubeSigmaPiActivation + subCubeActivation / (2 ^ subCubeDimension * multiCubeSigmaPiWeightsMax);
    end
end
end