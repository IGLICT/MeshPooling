function recon_from_interpolation(reconfolder,groundtruthfolder,targetfolder)
% reconstruct interpolation obj
if nargin < 3
    targetfolder = reconfolder;
end
if ~exist(targetfolder,'dir')
    mkdir(targetfolder)
end
matlist = dir([reconfolder,'\*.h5']);
for i = 1:size(matlist,1)
    [~,recon_name,~] = fileparts(matlist(i).name);
    folder = [targetfolder,'\',recon_name];
    if ~exist(folder, 'dir')
        mkdir(folder)
    end
    copyfile([reconfolder,'\',matlist(i).name],[folder,'\',matlist(i).name]);
    recon_from_random([groundtruthfolder,'\1.obj'],folder,folder);
end

end