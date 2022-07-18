function trainingDataSet = convertAndNormalizeTrainTestExcelFilesToDataset(trainingSetFile, testSetFile, inputs, outputs, datasetType)
% trainingSetFile: The training set file name
% testSetFile: The test set file name
% inputs: The number of inputs
% outputs: The number of outputs
% datasetType: The problem type - 0 for regression
%                               - 1 for classification

% Add project's folders to MATLAB path
loadPath;

num = [{xlsread(trainingSetFile)}, {xlsread(testSetFile)}];

% Check for non numeric values
for currentDataset = 1:size(num, 2)
    summatedNumTable = sum(sum(num{currentDataset}));
    if isnan(summatedNumTable)
        error('The dataset contains NaN values.');
    end
end

% Check for invalid number of outputs and dataset type combination
if outputs > 1 && datasetType == 1
    error('Invalid number of outputs and dataset type combination.');
end

% Create the datasets' header
header = cell(1, inputs + outputs);
for i = 1:inputs
    header{i} = sprintf('x%d', i);
end
for i = inputs + 1:inputs + outputs
    header{i} = sprintf('y%d', i - inputs);
end

% Add the header to each dataset
trainingDataSet = dataset({num{1}, header{:}});
testDataSet = dataset({num{2}, header{:}});

% Get the file name for each dataset file
[~, trainSetfileName] = fileparts(trainingSetFile);
[~, testSetfileName] = fileparts(testSetFile);

% Create the normalized training dataset
rowsNo = size(trainingDataSet, 1);
columnsNo = size(trainingDataSet, 2);
if datasetType == 1
    if min(double(trainingDataSet(:, end))) == 0
        trainingDataSet(:, end) = mat2dataset(double(trainingDataSet(:, end)) + 1);
    end
    columnsNo = columnsNo - 1;
end
xMax = max(abs(num{1}));
for i = 1:rowsNo
    for j = 1:columnsNo
        trainingDataSet{i, j} = num{1}(i, j) / xMax(j);
    end
end

% Create the normalized test dataset
rowsNo = size(testDataSet, 1);
columnsNo = size(testDataSet, 2);
if datasetType == 1
    if min(double(testDataSet(:, end))) == 0
        testDataSet(:, end) = mat2dataset(double(testDataSet(:, end)) + 1);
    end
    columnsNo = columnsNo - 1;
end
xMax = max(abs(num{2}));
for i = 1:rowsNo
    for j = 1:columnsNo
        testDataSet{i, j} = num{2}(i, j) / xMax(j);
    end
end

% Save both datasets
save(sprintf('datasets/%s.mat', trainSetfileName), 'trainingDataSet', 'datasetType', '-v7.3');
save(sprintf('datasets/%s.mat', testSetfileName), 'testDataSet', 'datasetType', '-v7.3');

% Remove project's folders from MATLAB path
unloadPath;

fprintf('Job Done.\n\n');
end

