function batch_rename(sourcefolder, targetfolder, suffix, del)
% rename file name to natural number

if nargin == 1
    targetfolder = [sourcefolder,'\rename'];
end
if nargin < 3
    suffix = 'obj';
end
if nargin < 4
    del = 0;
end
if ~exist(targetfolder, 'dir')
    mkdir(targetfolder);
end
filelist = dir([sourcefolder, '\*.', suffix]);
[~, id] = sort_nat({filelist.name});
filelist = filelist(id);
len = length(filelist);
for i = 1:len
    oldname = filelist(i).name;
    newname = [num2str(i),'.',suffix];
    status = copyfile([sourcefolder,'\',oldname],[targetfolder,'\',newname]);
    if status == 1
        disp([oldname, ' is renamed as ', newname])
    else
        disp([oldname, ' failed!'])
    end
    if del
        delete([sourcefolder,'\',oldname]);
    end
end