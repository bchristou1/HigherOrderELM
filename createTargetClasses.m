function [outputTable, classesNo] = createTargetClasses(inputTable)
classesNo = max(inputTable);
samplesNo = length(inputTable);
outputTable(1:samplesNo, 1:classesNo) = -1;
for i = 1:samplesNo
    outputTable(i, inputTable(i)) = 1;
end

