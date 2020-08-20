% demo reconstruct script
groundtruthfolder = '.\dense_scape\rename';
targetfolder = 'Q:\meshpooling\release\dense_scape_39.0_3.0_meanpooling_0.5_K=3';
file_name = [targetfolder, '\result.txt'];
data = dlmread(file_name);
m = size(data,1);
for j = 1:m
    if data(j,1) == 1
        epoch_num = j * 100;
        rebuild_loss = data(j,2);
        valid_loss = data(j,4);
    end
end
erms = recon_from_convvertex2(groundtruthfolder,targetfolder,epoch_num);
