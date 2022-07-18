function createKFoldCrossValidationDataSet(folds, inputDataSet, datasetType, filename)
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
filename = sprintf('%s_%dFoldsCV_2Sets', filename, folds);

% Check if the file exists and delete it
if exist(filename, 'file')
    delete(filename);
end

% Create the headers
dataSetHeaderTraining = cell(1, size(inputDataSet(1, :), 2));
dataSetHeaderTest = cell(1, size(inputDataSet(1, :), 2));
dataSetHeaderSize = size(dataSetHeaderTraining, 2);
for i = 1:dataSetHeaderSize
    dataSetHeaderTraining(i)  = cellstr(strcat(char(inputDataSet.Properties.VarNames(i)), 'Training'));
    dataSetHeaderTest(i)  = cellstr(strcat(char(inputDataSet.Properties.VarNames(i)), 'Test'));
end

% Convert the dataset to array
numericDataSet = double(inputDataSet(1:end, :));

% Shuffle the dataset
numericDataSet = numericDataSet(randperm(size(numericDataSet, 1)), :);

% Split the dataset
splittedDataSet = splitTrainingSet(folds, inputsNo, numericDataSet, false);

% Create the folds header for the dataset
dataSetFoldsHeader = cell(1,folds);
for i = 1:folds
    dataSetFoldsHeader(i) = {sprintf('fold%d', i)};
end

% Add the folds header to the datasets
trainingSetCV = dataset({cell(1, folds), dataSetFoldsHeader{:}});
testSetCV = dataset({cell(1, folds), dataSetFoldsHeader{:}});

% Create the training and test data for each fold
for i = 1:folds
    test = [splittedDataSet(1, i); splittedDataSet(2, i)];
    test = num2cell(cell2mat(test))';
    if i == 1
        training = [splittedDataSet(1, i + 1:end); splittedDataSet(2, i + 1:end)];
        training = num2cell(cell2mat(training))';
    else
        temp = [];
        training = [[splittedDataSet(1, 1:i - 1) splittedDataSet(1, i + 1:end)]; ...
            [splittedDataSet(2, 1:i - 1) splittedDataSet(2, i + 1:end)]];
        training = num2cell(cell2mat(training))';
    end
    trainingSetCV{1, i} = dataset({training, dataSetHeaderTraining{:}});
    testSetCV{1, i} = dataset({test, dataSetHeaderTest{:}});
end

% Save the data
save(sprintf('datasets/%s', filename), 'trainingSetCV', 'testSetCV', 'datasetType', '-v7.3');

% Remove project's folders from MATLAB path
unloadPath;

disp('File Successfully Created.')
end

