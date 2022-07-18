function [inputsNo, outputsNo] = findInputsOutputsNo(dataSetHeader)

inputsNo = 0;
outputsNo = 0;

for currentHeader = 1:size(dataSetHeader, 2)
    if strcmp(dataSetHeader{currentHeader}(1), 'x')
        inputsNo = inputsNo + 1;
    else
        outputsNo = outputsNo + 1;
    end
end

