close all;
clear;
clc;

% Load the dataset and set the parameters for the experiments
loadPath; % Add project's folders to MATLAB path
inputDatasetName = 'averageLocalizationErrorInSensorNodeLocalizationProcessReg_5FoldsCV_2Sets'; % The input dataset name without the file extension
load(inputDatasetName); % The input dataset
if exist('trainingSetCV','var') && exist('testSetCV','var')
    [inputsNo, outputsNo]  = findInputsOutputsNo(trainingSetCV{1, 1}.Properties.VarNames); % The number of neural network inputs and outputs
    entriesNo = size(trainingSetCV{1, 1}, 1) + size(testSetCV{1, 1}, 1);
    if datasetType == 1
        classesNo = max(dataset2matrix(testSetCV{1, 1}(:, end)));
    end
    foldsNo = size(trainingSetCV, 2); % The number of folds
    trainingSet = zeros(1, foldsNo);
    validationSetCV = zeros(1, foldsNo);
    testSet = zeros(1, foldsNo);
    datasetFormat = '2setsCV';
else
    if exist('trainingSet','var') && exist('testSet','var')
        [inputsNo, outputsNo]  = findInputsOutputsNo(trainingSet.Properties.VarNames); % The number of neural network inputs and outputs
        entriesNo = size(trainingSet, 1) + size(testSet, 1);
        if datasetType == 1
            classesNo = max(dataset2matrix(testSet(:, end)));
        end
        foldsNo = 1; % The number of folds
        trainingSetCV = [];
        validationSetCV = [];
        testSetCV = [];
        datasetFormat = '2sets';
    else
        [inputsNo, outputsNo]  = findInputsOutputsNo(trainingSetCV{1, 1}.Properties.VarNames); % The number of neural network inputs and outputs
        entriesNo = size(trainingSetCV{1, 1}, 1) + size(validationSetCV{1, 1}, 1) + size(testSet, 1);
        if datasetType == 1
            classesNo = max(dataset2matrix(testSet(:, end)));
        end
        foldsNo = size(trainingSetCV, 2); % The number of folds
        trainingSet = zeros(1, foldsNo);
        testSetCV = zeros(1, foldsNo);
        datasetFormat = '3sets';
    end
end

% User defined parameters
inputRange = [-1 1]; % The weight initialization range (e.g. [-1, 1])
networkTypes = {'loworder_loworder', ...
            'loworder_higherorder', ...
            'higherorder_loworder', ...
            'higherorder_higherorder', ...
            'loworder_multicube', ...
            'multicube_loworder', ...
            'multicube_multicube'};  % The matrix containing the network types that are going to be constructed - 'loworder_loworder' contains low-order neurons in the hidden layer and output layer
        %                                                                                                       - 'loworder_higherorder' contains low-order neurons in the hidden layer and higher-order neurons in the output layer
        %                                                                                                       - 'higherorder_loworder' contains higher-order neurons in the hidden layer and low-order neurons in the output layer
        %                                                                                                       - 'higherorder_higherorder' contains higher-order neurons in the hidden layer and higher-order neurons in the output layer
        %                                                                                                       - 'loworder_multicube' contains low-order neurons in the hidden layer and multi-cube neurons in the output layer
        %                                                                                                       - 'multicube_loworder' contains multi-cube neurons in the hidden layer and low-order neurons in the output layer
        %                                                                                                       - 'multicube_multicube' contains multi-cube neurons in the hidden layer and multi-cube neurons in the output layer
hiddenNodesNo = 10; % The number of hidden layer nodes
transferFunction = 'sig'; % The transfer function type - 'sig' for sigmoid,
%                                                      - 'sin' for sinusoid
%                                                      - 'tribas' for for triangular basis function
%                                                      - 'radbas' for radial basis function (for additive type of SLFNs instead of RBF type of SLFNs)
experimentsNo = 10; % The number of experiment runs

if sum(contains(networkTypes, 'multicube')) > 0
    multiCubeInputIds = ones(1, inputsNo); % The input dimension of each sub-unit for each input node
    multiCubeOutputIds = ones(1, hiddenNodesNo); % The input dimension of each sub-unit for each output node
end

% Automatically defined data
networkTypesNo = size(networkTypes, 2);
foldsHeaders = constructCustomHeader(1:foldsNo, 'fold');
experimentsHeaders = constructCustomHeader(1:experimentsNo, 'experiment');

testResults = dataset({cell(1, experimentsNo), experimentsHeaders{:}});
for currentExperiment = 1:experimentsNo
    testResults{1, currentExperiment} = dataset({cell(1, networkTypesNo), networkTypes{:}});
    if strcmp(datasetFormat, '2setsCV') || strcmp(datasetFormat, '3sets')
        for currentNetworkType = 1:networkTypesNo
            testResults{1, currentExperiment}{1, currentNetworkType} = dataset({cell(1, foldsNo), foldsHeaders{:}});
        end
    end
end

if strcmp(datasetFormat, '3sets')
    validationResults = dataset({cell(1, experimentsNo), experimentsHeaders{:}});
    for currentExperiment = 1:experimentsNo
        validationResults{1, currentExperiment} = dataset({cell(1, networkTypesNo), networkTypes{:}});
        if strcmp(datasetFormat, '2setsCV') || strcmp(datasetFormat, '3sets')
            for currentNetworkType = 1:networkTypesNo
                validationResults{1, currentExperiment}{1, currentNetworkType} = dataset({cell(1, foldsNo), foldsHeaders{:}});
            end
        end
    end
end

% Start the parallel pool
parallelPool = gcp('nocreate');
if isempty(parallelPool)
    parallelPool = parpool('local', feature('numCores'));
end

% Run the experiments
parfor currentExperiment = 1:experimentsNo
    for currentNetworkType = 1:networkTypesNo
        if strcmp(datasetFormat, '2sets')
            inputTrain = dataset2matrix(trainingSet(:, 1:inputsNo));
            outputTrain = dataset2matrix(trainingSet(:, inputsNo + 1:inputsNo + outputsNo));
            inputTest = dataset2matrix(testSet(:, 1:inputsNo));
            outputTest = dataset2matrix(testSet(:, inputsNo + 1:inputsNo + outputsNo));
            
            if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'loworder_loworder')
                testResults{1, currentExperiment}{1, currentNetworkType} = ELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, hiddenNodesNo, transferFunction, datasetType);
            end
            if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'loworder_higherorder')
                testResults{1, currentExperiment}{1, currentNetworkType} = sigmaSigmaPiELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, hiddenNodesNo, transferFunction, datasetType);
            end
            if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'higherorder_loworder')
                testResults{1, currentExperiment}{1, currentNetworkType} = sigmaPiSigmaELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, hiddenNodesNo, transferFunction, datasetType);
            end
            if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'higherorder_higherorder')
                testResults{1, currentExperiment}{1, currentNetworkType} = sigmaPiSigmaPiELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, hiddenNodesNo, transferFunction, datasetType);
            end
            if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'loworder_multicube')
                testResults{1, currentExperiment}{1, currentNetworkType} = sigmaMultiCubeELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, multiCubeInputIds, multiCubeOutputIds, hiddenNodesNo, transferFunction, datasetType);
            end
            if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'multicube_loworder')
                testResults{1, currentExperiment}{1, currentNetworkType} = multiCubeSigmaELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, multiCubeInputIds, multiCubeOutputIds, hiddenNodesNo, transferFunction, datasetType);
            end
            if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'multicube_multicube')
                testResults{1, currentExperiment}{1, currentNetworkType} = multiCubeMultiCubeELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, multiCubeInputIds, multiCubeOutputIds, hiddenNodesNo, transferFunction, datasetType);
            end
            fprintf('Experiment %d/%d Network Type %d/%d.\n', currentExperiment, experimentsNo, currentNetworkType, networkTypesNo);
        end
        if strcmp(datasetFormat, '2setsCV') || strcmp(datasetFormat, '3sets')
            for currentFold = 1:foldsNo
                inputTrain = dataset2matrix(trainingSetCV{1, currentFold}(:, 1:inputsNo));
                outputTrain = dataset2matrix(trainingSetCV{1, currentFold}(:, inputsNo + 1:inputsNo + outputsNo));
                
                if strcmp(datasetFormat, '2setsCV')
                    inputTest = dataset2matrix(testSetCV{1, currentFold}(:, 1:inputsNo));
                    outputTest = dataset2matrix(testSetCV{1, currentFold}(:, inputsNo + 1:inputsNo + outputsNo));
                    
                    if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'loworder_loworder')
                        testResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = ELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, hiddenNodesNo, transferFunction, datasetType);
                    end
                    if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'loworder_higherorder')
                        testResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = sigmaSigmaPiELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, hiddenNodesNo, transferFunction, datasetType);
                    end
                    if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'higherorder_loworder')
                        testResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = sigmaPiSigmaELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, hiddenNodesNo, transferFunction, datasetType);
                    end
                    if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'higherorder_higherorder')
                        testResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = sigmaPiSigmaPiELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, hiddenNodesNo, transferFunction, datasetType);
                    end
                    if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'loworder_multicube')
                        testResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = sigmaMultiCubeELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, multiCubeInputIds, multiCubeOutputIds, hiddenNodesNo, transferFunction, datasetType);
                    end
                    if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'multicube_loworder')
                        testResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = multiCubeSigmaELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, multiCubeInputIds, multiCubeOutputIds, hiddenNodesNo, transferFunction, datasetType);
                    end
                    if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'multicube_multicube')
                        testResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = multiCubeMultiCubeELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, multiCubeInputIds, multiCubeOutputIds, hiddenNodesNo, transferFunction, datasetType);
                    end
                    fprintf('Experiment %d/%d Network Type %d/%d Fold %d/%d.\n', currentExperiment, experimentsNo, currentNetworkType, networkTypesNo, currentFold, foldsNo);
                else
                    inputValidation = dataset2matrix(validationSetCV{1, currentFold}(:, 1:inputsNo));
                    outputValidation = dataset2matrix(validationSetCV{1, currentFold}(:, inputsNo + 1:inputsNo + outputsNo));
                    inputTest = dataset2matrix(testSet(:, 1:inputsNo));
                    outputTest = dataset2matrix(testSet(:, inputsNo + 1:inputsNo + outputsNo));
                    
                    if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'loworder_loworder')
                        validationResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = ELM(inputTrain, inputValidation, outputTrain, outputValidation, inputRange, hiddenNodesNo, transferFunction, datasetType);
                        testResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = ELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, hiddenNodesNo, transferFunction, datasetType);
                    end
                    if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'loworder_higherorder')
                        validationResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = sigmaSigmaPiELM(inputTrain, inputValidation, outputTrain, outputValidation, inputRange, hiddenNodesNo, transferFunction, datasetType);
                        testResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = sigmaSigmaPiELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, hiddenNodesNo, transferFunction, datasetType);
                    end
                    if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'higherorder_loworder')
                        validationResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = sigmaPiSigmaELM(inputTrain, inputValidation, outputTrain, outputValidation, inputRange, hiddenNodesNo, transferFunction, datasetType);
                        testResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = sigmaPiSigmaELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, hiddenNodesNo, transferFunction, datasetType);
                    end
                    if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'higherorder_higherorder')
                        validationResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = sigmaPiSigmaPiELM(inputTrain, inputValidation, outputTrain, outputValidation, inputRange, hiddenNodesNo, transferFunction, datasetType);
                        testResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = sigmaPiSigmaPiELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, hiddenNodesNo, transferFunction, datasetType);
                    end
                    if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'loworder_multicube')
                        validationResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = sigmaMultiCubeELM(inputTrain, inputValidation, outputTrain, outputValidation, inputRange, multiCubeInputIds, multiCubeOutputIds, hiddenNodesNo, transferFunction, datasetType);
                        testResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = sigmaMultiCubeELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, multiCubeInputIds, multiCubeOutputIds, hiddenNodesNo, transferFunction, datasetType);
                    end
                    if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'multicube_loworder')
                        validationResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = multiCubeSigmaELM(inputTrain, inputValidation, outputTrain, outputValidation, inputRange, multiCubeInputIds, multiCubeOutputIds, hiddenNodesNo, transferFunction, datasetType);
                        testResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = multiCubeSigmaELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, multiCubeInputIds, multiCubeOutputIds, hiddenNodesNo, transferFunction, datasetType);
                    end
                    if strcmp(testResults{1, currentExperiment}(1, currentNetworkType).Properties.VarNames, 'multicube_multicube')
                        validationResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = multiCubeMultiCubeELM(inputTrain, inputValidation, outputTrain, outputValidation, inputRange, multiCubeInputIds, multiCubeOutputIds, hiddenNodesNo, transferFunction, datasetType);
                        testResults{1, currentExperiment}{1, currentNetworkType}{1, currentFold} = multiCubeMultiCubeELM(inputTrain, inputTest, outputTrain, outputTest, inputRange, multiCubeInputIds, multiCubeOutputIds, hiddenNodesNo, transferFunction, datasetType);
                    end
                    fprintf('Experiment %d/%d Network Type %d/%d Fold %d/%d.\n', currentExperiment, experimentsNo, currentNetworkType, networkTypesNo, currentFold, foldsNo);
                end
            end
        end
    end
end

% Shut down the parallel pool
parallelPool = gcp('nocreate');
if ~isempty(parallelPool)
    delete(parallelPool);
end

% Save the experimental results
outputResultsFilename = strcat(inputDatasetName(1:end), 'OutputResults.mat');
if strcmp(datasetFormat, '3sets')
    if datasetType == 1
        save(sprintf('results/%s', outputResultsFilename), 'validationResults', 'testResults', 'hiddenNodesNo', 'datasetType', 'datasetFormat', 'entriesNo', 'inputsNo', 'outputsNo', 'classesNo', 'inputDatasetName', '-v7.3');
    else
        save(sprintf('results/%s', outputResultsFilename), 'validationResults', 'testResults', 'hiddenNodesNo', 'datasetType', 'datasetFormat', 'entriesNo', 'inputsNo', 'outputsNo', 'inputDatasetName', '-v7.3');
    end
else
    if datasetType == 1
        save(sprintf('results/%s', outputResultsFilename), 'testResults', 'hiddenNodesNo', 'datasetType', 'datasetFormat', 'entriesNo', 'inputsNo', 'outputsNo', 'classesNo', 'inputDatasetName', '-v7.3');
    else
        save(sprintf('results/%s', outputResultsFilename), 'testResults', 'hiddenNodesNo', 'datasetType', 'datasetFormat', 'entriesNo', 'inputsNo', 'outputsNo', 'inputDatasetName', '-v7.3');
    end
end

% Display the experimental results
displayResults(outputResultsFilename);

% Remove project's folders from MATLAB path
unloadPath;