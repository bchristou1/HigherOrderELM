function createKFoldCrossValidationDataSet2SetsAnd3Sets(testSetPercentage, folds, inputDataSet, datasetType, filename)
% testSetPercentage: The test set percentage
% folds: The number of folds
% inputDataSet: The name of the input dataset in MATLAB dataset format
% datasetType: The problem type - 0 for regression
%                               - 1 for classification
% filename: The output file name

% Add project's folders to MATLAB path
loadPath;

clc;
warning('off');
inputsNo = findInputsOutputsNo(inputDataSet.Properties.VarNames);
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
dataSetHeaderValidation = cell(1, size(inputDataSet(1, :), 2));
dataSetHeaderTraining = cell(1, size(inputDataSet(1, :), 2));
dataSetHeaderTest = cell(1, size(inputDataSet(1, :), 2));
dataSetHeaderSize = size(dataSetHeaderTraining, 2);

for i = 1:dataSetHeaderSize 
    dataSetHeaderTraining(i)  = cellstr(strcat(char(inputDataSet.Properties.VarNames(i)), 'Training'));
    dataSetHeaderValidation(i)  = cellstr(strcat(char(inputDataSet.Properties.VarNames(i)), 'Validation'));
    dataSetHeaderTest(i)  = cellstr(strcat(char(inputDataSet.Properties.VarNames(i)), 'Test'));
end

% Convert the dataset to array
numericDataSet = double(inputDataSet(1:end, :));

% Shuffle the dataset
numericDataSet = numericDataSet(randperm(size(numericDataSet,1)), :);
numericDataSet = numericDataSet(randperm(size(numericDataSet,1)), :);

% Create the test set
samplesNo = size(inputDataSet(:, 1), 1);
testSetSize = round(samplesNo * testSetPercentage);
testSetRowsNo = datasample(RandStream('mlfg6331_64'), 1:samplesNo, testSetSize, 'Replace', false);
testSet = zeros(testSetSize, dataSetHeaderSize);
tempNumericDataset = zeros(samplesNo - testSetSize, dataSetHeaderSize);
count = 0;
testSetCount = 0;
for i = 1:samplesNo
    if ismember(i, testSetRowsNo)
        testSetCount = testSetCount + 1;
        testSet(testSetCount, :) = numericDataSet(testSetRowsNo(testSetCount), :);
    else
        count = count + 1;
        tempNumericDataset(count, :) = numericDataSet(i, :);
    end
end
numericDataSet = tempNumericDataset;
clear tempNumericDataset;

% Add the header to the test dataset
testSet = dataset({testSet, dataSetHeaderTest{:}});

% Shuffle the dataset
numericDataSet = numericDataSet(randperm(size(numericDataSet,1)), :);
numericDataSet = numericDataSet(randperm(size(numericDataSet,1)), :);

% Split the training dataset
splittedDataSet = splitTrainingSet(folds, inputsNo, numericDataSet, false);

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

