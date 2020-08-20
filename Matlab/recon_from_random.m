function [latent_z]=recon_from_random(firstfile,matfolder,workpath,name)
% point feature
if nargin<3
    workpath=[matfolder,'\..\mesha'];
    name='test_mesh';
elseif nargin<4
     name='test_mesh';   
end
if ~exist(workpath,'file')
    mkdir(workpath);
end
originfile = firstfile;
if ischar(matfolder)
    matlist=dir([matfolder,'\*.h5']);
else
    matlist=1;
end
NLOGDR=[];
NS=[];
latent_z=[];
for i=1:size(matlist,1)
    NLOGDR=[];
    NS=[];
    if ischar(matfolder)
        m.test_mesh = h5read([matfolder,'\',matlist(i).name],['/',name]);
        gen=permute(m.test_mesh,[3,1,2]);
        FLOGDR=gen(:,1:3,:);
        if size(gen,2)>3
            FS=gen(:,4:9,:);
        else
            lowbarindex=strfind(matlist(i).name,'_');
            if (ismember('A',matlist(i).name)||ismember('a',matlist(i).name))
               [mat_namea,~]=searchmat(matfolder);
            end
            if (ismember('B',matlist(i).name)||ismember('b',matlist(i).name))
               [~,mat_namea]=searchmat(matfolder);
            end
            mat=load(mat_namea);
            FS=permute(mat.FS,[1,3,2]);
            if size(FS,1)~=size(FLOGDR,1)||(ismember('a',matlist(i).name)&&ismember('b',matlist(i).name))
            FS=zeros(size(FLOGDR).*[1,2,1]);
            FS(:,[1,4,6],:)=ones(size(FLOGDR));
            end
        end
        
        FLOGDR=reshape(FLOGDR,size(FLOGDR,1),size(FLOGDR,2)*size(FLOGDR,3));
        FS=reshape(FS,size(FS,1),size(FS,2)*size(FS,3));
    else
        gen=matfolder;
        m.latent_z=[];
        FLOGDR=gen(:,:,1:3);
        FS=gen(:,:,4:9);
    end
    
    [ NLOGDR1, NS1 ] = InverseMap(FLOGDR,FS);
    NLOGDR=[NLOGDR;NLOGDR1];
    NS=[NS;NS1];
end

for i=1:size(NS,1)
    ACAP(originfile,[workpath,'\',sprintf('%05d',i),name,'.obj'],NLOGDR(i,:), NS(i,:));
end

end