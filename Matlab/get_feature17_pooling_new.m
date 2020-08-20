function get_feature17_pooling_new(densefolder,simmesh,tree_vector,matpath)
% get ACAP feature with pooling and the convolution
% FLOGRNEW,FS,neighbour1,,neighbour2,mapping,demapping
if nargin < 4
    matpath = densefolder;
end
if ~exist(matpath, 'dir')
    mkdir(matpath)
end

vdensemeshlist = dir([densefolder,'\*.obj']);
[~,i] = sort_nat({vdensemeshlist.name});
vdensemeshlist = vdensemeshlist(i);
if ~exist([densefolder,'\fv_r.mat'],'file')
    % more mesh models, more memory needed!
    GetFeatureR(densefolder,length(vdensemeshlist))
end
fv = load([densefolder,'\fv_r.mat']);

[vdensemesh,~,~,~,~,VVdense,cotweight_dense]=cotlp([densefolder,'\',vdensemeshlist(1).name]);
if nargin < 2
    simmesh = [densefolder,'\',vdensemeshlist(1).name];
end
[vsimpmesh,~,~,~,~,VVsimp,cotweight_simp]=cotlp(simmesh);

W1 = full(cotweight_dense);
W2 = full(cotweight_simp);
for i = 1:size(W1,1)
    for j = 1:size(W1,2)
        if W1(i,j) ~= 0
            W1(i,j)=1;
        end
    end
end
for i = 1:size(W2,1)
    for j = 1:size(W2,2)
        if W2(i,j) ~= 0
            W2(i,j)=1;
        end
    end
end

neighbour1=zeros(size(vdensemesh,1),100);
neighbour2=zeros(size(vsimpmesh,1),100);
maxnum=0;
for i=1:size(VVdense,1)
    neighbour1(i,1:size(VVdense{i,:},2))=VVdense{i,:};
    if size(VVdense{i,:},2)>maxnum
        maxnum=size(VVdense{i,:},2);
    end
end
neighbour1(:,maxnum+1:end)=[];
maxnum=0;
for i=1:size(VVsimp,1)
    neighbour2(i,1:size(VVsimp{i,:},2))=VVsimp{i,:};
    if size(VVsimp{i,:},2)>maxnum
        maxnum=size(VVsimp{i,:},2);
    end
end
neighbour2(:,maxnum+1:end)=[];

vdensenum = size(vdensemesh, 1);
mapping = tree2mapping(tree_vector, vdensenum);


[fmlogdr, fms] = FeatureMap(fv.LOGRNEW, fv.S);
feature = cat(2, fms, fmlogdr);
fmlogdr=permute(reshape(fmlogdr,size(fmlogdr,1),3,size(vdensemesh,1)),[1,3,2]);
fms=permute(reshape(fms,size(fms,1),6,size(vdensemesh,1)),[1,3,2]);
cotweight1=zeros(size(neighbour1));
cotweight2=zeros(size(neighbour2));
for i=1:size(neighbour1,1)
    for j=1:size(neighbour1,2)
        if neighbour1(i,j)>0
%                 cotweight1(i,j)=cotweight_dense(i,neighbour1(i,j));
            cotweight1(i,j)=1/length(nonzeros(neighbour1(i,:)));
        end
    end
end
for i=1:size(neighbour2,1)
    for j=1:size(neighbour2,2)
        if neighbour2(i,j)>0
%                 cotweight2(i,j)=cotweight_simp(i,neighbour2(i,j));
            cotweight2(i,j)=1/length(nonzeros(neighbour2(i,:)));
        end
    end
end
iii=1:size(fmlogdr,1);
FLOGRNEW=fmlogdr(iii,:,:);
FS=fms(iii,:,:);
neighbour=neighbour1;
neighbour1=neighbour1;
neighbour2=neighbour2;
mapping=mapping;
feature=feature(iii,:,:);
cotweight1=cotweight1;
cotweight2=cotweight2;

save([matpath,'\vertFeaturepoolingc.mat'],'FLOGRNEW','FS','neighbour',...
    'neighbour1','neighbour2','mapping','feature','cotweight1','cotweight2','W1','W2','-v7.3')

end
