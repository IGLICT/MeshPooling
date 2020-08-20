% demo preprocessing script for SCAPE
data_folder = '.\dense_scape';

% rename obj
dense_folder = fullfile(data_folder,'rename');
batch_rename(data_folder, dense_folder)

% edge contraction
simfolder = [dense_folder,'\sim'];
simratio = 0.5;
simtype = 1;
sim_gamma = 0.005;
if ~exist(simfolder,'dir')
    mkdir(simfolder);
end
oriobj = fullfile(dense_folder, '1.obj');
simobj = fullfile(simfolder,['loss',num2str(simtype),'_',num2str(sim_gamma),'.obj']);
[tree,v1,~] = MeshSimp(oriobj, simobj, simratio, simtype, sim_gamma);
v2 = cotlp(simobj);
assert(size(v1,2)==size(v2,1));    % check simplification succeed or not

% ACAP feature preparation
matpath = [dense_folder,'\feature'];
get_feature17_pooling_new(dense_folder, simobj, tree, matpath);
