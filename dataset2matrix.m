function output = dataset2matrix(dataSet)
variableNames = dataSet.Properties.VarNames;
dataSet = dataset2cell(dataSet);
emptyCells = cellfun('isempty', dataSet); 
dataSet(all(emptyCells,2),:) = [];
if strcmp(variableNames{1, 1}, dataSet{1, 1})
    dataSet = dataSet(2:end, :);
end
output = zeros(size(dataSet, 1), size(dataSet, 2));
if size(dataSet, 1) * size(dataSet, 2) == size(dataSet, 1) || size(dataSet, 1) * size(dataSet, 2) == size(dataSet, 2)
    [x, ~] = size(dataSet);
    for i = 1:length(dataSet)
        if x == 1        
            output(i) = dataSet{1, i};
        else
            output(i) = dataSet{i, 1};
        end
    end
else
    for i = 1:size(dataSet, 1)
        for j = 1:size(dataSet, 2)
            output(i, j) = dataSet{i, j};
        end
    end
end
end

