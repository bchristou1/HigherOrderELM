function [output] = splitTrainingSet(folds, inputsNo, trainingSet, removeHeaders)

if removeHeaders == true
    trainingSet = trainingSet(2:end, :);
end

count = 0;
trainingSetSize = size(trainingSet, 1);
tiles = zeros(1, folds);
initialValues = floor(trainingSetSize / folds);
outputsNo = size(trainingSet, 2) - inputsNo;

for currentFold = 1:folds
    tiles(currentFold) = initialValues;
end

diff = trainingSetSize - initialValues * folds;

if mod(trainingSetSize, folds) ~= 0
    for currentFold = 1:folds
        count = count + 1;
        tiles(currentFold) = tiles(currentFold) + 1;
        if count == diff
            break
        end
    end
end

output = [];
sum = 0;

for currentFold = 1:folds
    row = [];
    tempRow = [];
    sum = sum + tiles(currentFold);
    for currentInput = 1:inputsNo
        tempRow = [tempRow; [trainingSet(sum - tiles(currentFold) + 1:sum, currentInput)']];
    end
    row = [row; tempRow];
    tempRow = [];
    for currentOutput = inputsNo + 1:size(trainingSet, 2)
        tempRow = [tempRow; [trainingSet(sum - tiles(currentFold) + 1:sum, currentOutput)']];
    end
    row = [row; {tempRow}];
    output = [output row];
end

end