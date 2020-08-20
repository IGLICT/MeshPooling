function erms = recon_from_convvertex2(groundtruthfolder,folder,step)
% recon models from h5
% randomh5=[folder,'\randomtest',num2str(step),'.h5'];
recon_h5=[folder,'\rebuild',num2str(step),'.h5'];
% [~,random_name]=fileparts(randomh5);
[~,recon_name]=fileparts(recon_h5);
% random_folder=[folder,'\',random_name];
recon_folder=[folder,'\',recon_name];

% if ~exist(random_folder,'dir')
%     mkdir(random_folder)
% end
if ~exist(recon_folder,'dir')
    mkdir(recon_folder)
end

% copyfile(randomah5,[random_folder,'\randomtest',num2str(step),'.h5'])
copyfile(recon_h5,[recon_folder,'\recon',num2str(step),'.h5'])
recon_from_random([groundtruthfolder,'\1.obj'],recon_folder,recon_folder);

valid_id = h5read(recon_h5,'/valid_id');
valid_id = valid_id + 1;
validtruthfolder = [groundtruthfolder, '\valid'];
validreconfolder = [recon_folder,'\valid'];
if ~exist(validtruthfolder, 'dir')
    mkdir(validtruthfolder);
end
if ~exist(validreconfolder, 'dir')
    mkdir(validreconfolder);
end
batch_rename(recon_folder,[recon_folder,'\rename']);
for i = 1:length(valid_id)
    copyfile([groundtruthfolder,'\',num2str(valid_id(i)),'.obj'],[validtruthfolder,'\',num2str(valid_id(i)),'.obj']);
    copyfile([recon_folder,'\rename\',num2str(valid_id(i)),'.obj'],[validreconfolder,'\',num2str(valid_id(i)),'.obj']);
end
erms = show_error(validtruthfolder,validreconfolder);
delete([recon_folder,'\rename\*.obj']);
rmdir([recon_folder,'\rename']);
% delete([validtruthfolder,'\*.obj']);
% rmdir(validtruthfolder);

end