function resultsTest = sigmaMultiCubeELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, multiCubeInputIds, multiCubeOutputIds, hiddenNodesNo, transferFunction, datasetType)

% inputTrain: The input training set
% inputTest: The input test set
% outputTrain: The output training set
% outputTest: The output test set
% inputRange: The weight initialization range (e.g. [-1, 1])
% multiCubeIds: The input dimensions of each sub-unit
% hiddenNodesNo: The number of hidden layer nodes
% transferFunction: The transfer function type - 'sig' for sigmoid,
%                                              - 'sin' for sinusoid
%                                              - 'tribas' for for triangular basis function
%                                              - 'radbas' for radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
%                                              - 'gaussian' for gaussian function
% type: The problem type - 0 for regression
%                        - 1 for classification

id = size(inputTrain, 2);
samplesNoTrain = size(inputTrain, 1);
samplesNoTest = size(inputTest, 1);
outputWeightsNo = sum(2 .^ hiddenNodesNo);

% Contruct the classification outputs matrices
if datasetType
    outputTrain = createTargetClasses(outputTrain);
end

% Make the necessary initializations
inputWeights = inputRange(1) + (inputRange(2)- inputRange(1)) .* randn(hiddenNodesNo, sum(2 .^ multiCubeInputIds));
inputThresholds = inputRange(1) + (inputRange(2)- inputRange(1)) .* randn(hiddenNodesNo, 1);
HTrain = zeros(samplesNoTrain, hiddenNodesNo);
HTest = zeros(samplesNoTest, hiddenNodesNo);
augSigmaPiHTrain = zeros(samplesNoTrain, outputWeightsNo);
augSigmaPiHTest = zeros(samplesNoTest, outputWeightsNo);
w = ones(1, outputWeightsNo);

% Calculate the hidden layer matrix H for the training set
for currentSample = 1:samplesNoTrain
    for currentHiddenNodeNo = 1:hiddenNodesNo
        if strcmp(transferFunction, 'sig') || strcmp(transferFunction, 'sin') || strcmp(transferFunction, 'tribas') || strcmp(transferFunction, 'radbas') || strcmp(transferFunction, 'gaussian')
            if strcmp(transferFunction, 'sig')
                HTrain(currentSample, currentHiddenNodeNo) = logsig(inputTrain(currentSample, :) * inputWeights(currentHiddenNodeNo, 1:id)' + inputThresholds(currentHiddenNodeNo));
            end
            if strcmp(transferFunction, 'sin')
                HTrain(currentSample, currentHiddenNodeNo) = sin(inputTrain(currentSample, :) * inputWeights(currentHiddenNodeNo, 1:id)' + inputThresholds(currentHiddenNodeNo));
            end
            if strcmp(transferFunction, 'tribas')
                HTrain(currentSample, currentHiddenNodeNo) = tribas(inputTrain(currentSample, :) * inputWeights(currentHiddenNodeNo, 1:id)' + inputThresholds(currentHiddenNodeNo));
            end
            if strcmp(transferFunction, 'radbas')
                HTrain(currentSample, currentHiddenNodeNo) = radbas(inputTrain(currentSample, :) * inputWeights(currentHiddenNodeNo, 1:id)' + inputThresholds(currentHiddenNodeNo));
            end
            if strcmp(transferFunction, 'gaussian')
                HTrain(currentSample, currentHiddenNodeNo) = gaussian(inputTrain(currentSample, :) * inputWeights(currentHiddenNodeNo, 1:id)' + inputThresholds(currentHiddenNodeNo));
            end
        else
            error('Invalid transfer function type');
        end
    end
end

% Calculate the hidden layer matrix H for the test set
for currentSample = 1:samplesNoTest
    for currentHiddenNodeNo = 1:hiddenNodesNo
        if strcmp(transferFunction, 'sig')
            HTest(currentSample, currentHiddenNodeNo) = logsig(inputTest(currentSample, :) * inputWeights(currentHiddenNodeNo, 1:id)' + inputThresholds(currentHiddenNodeNo));
        end
        if strcmp(transferFunction, 'sin')
            HTest(currentSample, currentHiddenNodeNo) = sin(inputTest(currentSample, :) * inputWeights(currentHiddenNodeNo, 1:id)' + inputThresholds(currentHiddenNodeNo));
        end
        if strcmp(transferFunction, 'tribas')
            HTest(currentSample, currentHiddenNodeNo) = tribas(inputTest(currentSample, :) * inputWeights(currentHiddenNodeNo, 1:id)' + inputThresholds(currentHiddenNodeNo));
        end
        if strcmp(transferFunction, 'radbas')
            HTest(currentSample, currentHiddenNodeNo) = radbas(inputTest(currentSample, :) * inputWeights(currentHiddenNodeNo, 1:id)' + inputThresholds(currentHiddenNodeNo));
        end
        if strcmp(transferFunction, 'gaussian')
            HTest(currentSample, currentHiddenNodeNo) = gaussian(inputTest(currentSample, :) * inputWeights(currentHiddenNodeNo, 1:id)' + inputThresholds(currentHiddenNodeNo));
        end
    end
end

% Create the augmented matrix for the training set
for currentSample = 1:samplesNoTrain
    [~, augSigmaPiHTrain(currentSample, :)] = multiCubeSigmaPiNeuron(HTrain(currentSample, :), w, multiCubeOutputIds);
end

% Create the augmented matrix for the test set
for currentSample = 1:samplesNoTest
    [~, augSigmaPiHTest(currentSample, :)] = multiCubeSigmaPiNeuron(HTest(currentSample, :), w, multiCubeOutputIds);
end


% Calculate the output neuron weights
outputWeights = pinv(augSigmaPiHTrain) * outputTrain;

% Calculate the test set network output
networkOutputTest = augSigmaPiHTest * outputWeights;

% Calculate the network accuracy
if datasetType == 0
    resultsTest = mean(mean((outputTest - networkOutputTest) .^ 2));
else
    if datasetType == 1
        errorsNoTest = 0;
        for currentSample = 1:samplesNoTest
            [~, currentClass] = max(networkOutputTest(currentSample, :));
            if currentClass ~= outputTest(currentSample)
                errorsNoTest =  errorsNoTest + 1;
            end
        end
        resultsTest = 1 - (errorsNoTest / samplesNoTest);
    else
        error('Invalid problem type');
    end
end
