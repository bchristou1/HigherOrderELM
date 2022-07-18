function dataSet = convertAndNormalizeExcelToDataset(file, inputs, outputs, datasetType)
% file: The file name
% inputs: The number of inputs
% outputs: The number of outputs
% datasetType: The problem type - 0 for regression
%                               - 1 for classification

% Add project's folders to MATLAB path
loadPath;

num = xlsread(file);

% Check for non numeric values
summatedNumTable = sum(sum(num));
if isnan(summatedNumTable)
    error('The dataset contains NaN values.');
end

% Check for invalid number of outputs and dataset type combination
if outputs > 1 && datasetType == 1
    error('Invalid number of outputs and dataset type combination.');
end

% Create the dataset's header
header = cell(1, inputs + outputs);
for i = 1:inputs
    header{i} = sprintf('x%d', i);
end
for i = inputs + 1:inputs + outputs
    header{i} = sprintf('y%d', i - inputs);
end

% Add the header to the dataset
dataSet = dataset({num, header{:}});


% Get the file name for the dataset file
[~, fileName] = fileparts(file);

% Create the normalized dataset
rowsNo = size(dataSet, 1);
columnsNo = size(dataSet, 2);
if datasetType == 1
    if min(double(dataSet(:, end))) == 0
        dataSet(:, end) = mat2dataset(double(dataSet(:, end)) + 1);
    end
    columnsNo = columnsNo - 1;
end
xMax = max(abs(num));
for i = 1:rowsNo
    for j = 1:columnsNo
        dataSet{i, j} = num(i, j) / xMax(j);
    end
end

% Save the dataset
save(sprintf('datasets/%s.mat', fileName), 'dataSet', 'datasetType', '-v7.3'); 

% Remove project's folders from MATLAB path
unloadPath; 

fprintf('Job Done.\n\n');
end

