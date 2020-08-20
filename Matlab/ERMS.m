function Erms=ERMS(groundtruthfolder,testfolder)

gtlist = dir([groundtruthfolder,'\*.obj']);
[~, i] = sort_nat({gtlist.name});
gtlist = gtlist(i);

testlist = dir([testfolder,'\*.obj']);
[~, i] = sort_nat({testlist.name});
testlist = testlist(i);

assert(size(gtlist,1)==size(testlist,1));

difference = zeros(size(testlist,1),1);
for i =1:size(gtlist,1)
    [rv]=cotlp([groundtruthfolder,'\',gtlist(i).name]);
    [ov]=cotlp([testfolder,'\',testlist(i).name]);
    difference(i) = sum(sum((rv-ov).^2));
end

Erms = 1000*sqrt(sum(difference))/sqrt(3*size(gtlist,1)*size(rv,1));
end