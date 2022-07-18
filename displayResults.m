function displayResults(filename)
% filename: The results file name

clc;
load(filename);

experimentsNo = size(testResults, 2);
networkTypesNo = size(testResults{1, 1}, 2);
networkTypesHeaders = testResults{1, 1}.Properties.VarNames;
experimentalValidationResults = zeros(experimentsNo, networkTypesNo);
experimentalTestResults = zeros(experimentsNo, networkTypesNo);

for currentExperiment = 1:experimentsNo
    for currentNetworkType = 1:networkTypesNo
        if strcmp(datasetFormat, '2setsCV') || strcmp(datasetFormat, '3sets')
            experimentalTestResults(currentExperiment, currentNetworkType) = mean(dataset2matrix(testResults{1, currentExperiment}{1, currentNetworkType}(1, :)));
            if strcmp(datasetFormat, '3sets')
                experimentalValidationResults(currentExperiment, currentNetworkType) = mean(dataset2matrix(validationResults{1, currentExperiment}{1, currentNetworkType}(1, :)));
            end
        else
            experimentalTestResults(currentExperiment, currentNetworkType) = dataset2matrix(testResults{1, currentExperiment}(1, currentNetworkType));
        end
        
    end
end

finalExperimentalValidationResults = mean(experimentalValidationResults);
finalExperimentalTestResults = mean(experimentalTestResults);

if datasetType == 1
    metric = 'Classification Accuracy';
    [~, bestValidationSetNetworkPosition] = max(finalExperimentalValidationResults);
    [~, bestTestSetNetworkPosition] = max(finalExperimentalTestResults);
else
    metric = 'Mean Square Error';
    [~, bestValidationSetNetworkPosition] = min(finalExperimentalValidationResults);
    [~, bestTestSetNetworkPosition] = min(finalExperimentalTestResults);
end

if strcmp(datasetFormat, '3sets')
    if datasetType == 1
        fprintf('Dataset: %s, Entries: %d, Attributes: %d, Classes: %d (Validation Set Results).\n', inputDatasetName, entriesNo, inputsNo + outputsNo, classesNo);
    else
        fprintf('Dataset: %s, Entries: %d, Attributes: %d (Validation Set Results).\n', inputDatasetName, entriesNo, inputsNo + outputsNo);
    end
    for currentNetworkType = 1:networkTypesNo
        if currentNetworkType == bestValidationSetNetworkPosition
            bestValidationSet = '(Best)';
        else
            bestValidationSet = '';
        end
        
        if datasetType == 1
            fprintf('Network Type: %s, Metric: %s, Validation Set Result: %.2f%% %s\n', networkTypesHeaders{currentNetworkType}, metric, finalExperimentalValidationResults(currentNetworkType) * 100, bestValidationSet);
        else
            fprintf('Network Type: %s, Metric: %s, Validation Set Result: %.6f %s\n', networkTypesHeaders{currentNetworkType}, metric, finalExperimentalValidationResults(currentNetworkType), bestValidationSet);
        end
    end
    fprintf('\n');
end

if datasetType == 1
    fprintf('Dataset: %s, Entries: %d, Attributes: %d, Classes: %d (Test Set Results).\n', inputDatasetName, entriesNo, inputsNo + outputsNo, classesNo);
else
    fprintf('Dataset: %s, Entries: %d, Attributes: %d (Test Set Results).\n', inputDatasetName, entriesNo, inputsNo + outputsNo);
end
for currentNetworkType = 1:networkTypesNo
    if currentNetworkType == bestTestSetNetworkPosition
        bestTestSet = '(Best)';
    else
        bestTestSet = '';
    end
    
    if datasetType == 1
        fprintf('Network Type: %s, Metric: %s, Test Set Result: %.2f%% %s\n', networkTypesHeaders{currentNetworkType}, metric, finalExperimentalTestResults(currentNetworkType) * 100, bestTestSet);
    else
        fprintf('Network Type: %s, Metric: %s, Test Set Result: %.6f %s\n', networkTypesHeaders{currentNetworkType}, metric, finalExperimentalTestResults(currentNetworkType), bestTestSet);
    end
end
fprintf('\nJob done.\n');

end