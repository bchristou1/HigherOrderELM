function createKFoldCrossValidationDataSetSeparateSets2SetsAnd3Sets(folds, trainingSet, testSet, datasetType, filename)
% folds: The number of folds
% trainingSet: The name of the input training dataset in MATLAB dataset format
% testSet: The name of the input test dataset in MATLAB dataset format
% filename: The output file name

% Add project's folders to MATLAB path
loadPath;

clc;
warning('off');
inputsNo = findInputsOutputsNo(testSet.Properties.VarNames);
filename2Sets = sprintf('%s_%dFolds_2Sets', filename, folds);
filename3Sets = sprintf('%s_%dFolds_3Sets', filename, folds);

% Check if files exist and delete them
if exist(filename2Sets, 'file')
    delete(filename2Sets);
end

if exist(filename3Sets, 'file')
    delete(filename3Sets);
end

% Create the headers
dataSetHeaderTraining = cell(1, size(testSet(1, :), 2));
dataSetHeaderValidation = cell(1, size(testSet(1, :), 2));
dataSetHeaderTest = cell(1, size(testSet(1, :), 2));
dataSetHeaderSize = size(dataSetHeaderTraining, 2);
for i = 1:dataSetHeaderSize 
    dataSetHeaderTraining(i) = cellstr(strcat(char(testSet.Properties.VarNames(i)), 'Training'));
    dataSetHeaderValidation(i) = cellstr(strcat(char(testSet.Properties.VarNames(i)), 'Validation'));
    dataSetHeaderTest(i)  = cellstr(strcat(char(testSet.Properties.VarNames(i)), 'Test'));
end

% Convert each dataset to array
numericSetTrain = double(trainingSet(1:end, :));
numericSetTest = double(testSet(1:end, :));

% Shuffle the datasets
numericSetTrain = numericSetTrain(randperm(size(numericSetTrain,1)), :);
numericSetTrain = numericSetTrain(randperm(size(numericSetTrain,1)), :);
numericSetTest = numericSetTest(randperm(size(numericSetTest,1)), :);
numericSetTest = numericSetTest(randperm(size(numericSetTest,1)), :);

% Add the header to the test dataset
testSet = dataset({numericSetTest, dataSetHeaderTest{:}});

% Split the training dataset
splittedDataSet = splitTrainingSet(folds, inputsNo, numericSetTrain, false);

% Create the folds header for the datasets
dataSetFoldsHeader = cell(1,folds);
for i = 1:folds
    dataSetFoldsHeader(i) = {sprintf('fold%d', i)};
end

% Add the folds header to the datasets
trainingSetCV = dataset({cell(1, folds), dataSetFoldsHeader{:}});
validationSetCV = dataset({cell(1, folds), dataSetFoldsHeader{:}});

% Create the training and validation data for each fold
for i = 1:folds
    validation = [splittedDataSet(1, i); splittedDataSet(2, i)];
    validation = num2cell(cell2mat(validation))';
    if i == 1
        training = [splittedDataSet(1, i + 1:end); splittedDataSet(2, i + 1:end)];
        training = num2cell(cell2mat(training))';
        trainingSet = dataset({[training; validation], dataSetHeaderTraining{:}});
    else
        training = [[splittedDataSet(1, 1:i - 1) splittedDataSet(1, i + 1:end)]; ...
            [splittedDataSet(2, 1:i - 1) splittedDataSet(2, i + 1:end)]];
        training = num2cell(cell2mat(training))';
    end
    trainingSetCV{1, i} = dataset({training, dataSetHeaderTraining{:}});
    validationSetCV{1, i} = dataset({validation, dataSetHeaderValidation{:}});
end

% Save the data
save(sprintf('datasets/%s', filename3Sets), 'trainingSetCV', 'validationSetCV', 'testSet', 'datasetType', '-v7.3');
save(sprintf('datasets/%s', filename2Sets), 'trainingSet', 'testSet', 'datasetType', '-v7.3');

% Remove project's folders from MATLAB path
unloadPath;

disp('Files Successfully Created.')
end

