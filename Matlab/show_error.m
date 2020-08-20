function Erms=show_error(groundtruthfolder,testfolder,align)
% show the error with the ground truth (RMSE)
if nargin < 3
    align = 1;
end

groundtruthlist=dir([groundtruthfolder,'\*.obj']);
[~,i]=sort_nat({groundtruthlist.name});
groundtruthlist=groundtruthlist(i);

testlist=dir([testfolder,'\*.obj']);
[~,i]=sort_nat({testlist.name});
testlist=testlist(i);

assert(length(groundtruthlist)==length(testlist))

modeli = zeros(length(groundtruthlist),1);
for i = 1:length(testlist)
    vground = cotlp([groundtruthfolder,'\',groundtruthlist(i).name]);
    vtest = cotlp([testfolder,'\',testlist(i).name]);
    if align
        v1_align=vground;
        v2_align=vtest;
        v1_align_center = mean(v1_align,1);
        v2_align_center = mean(v2_align,1);
        H = v2_align'*v1_align-size(v2_align,1)*v2_align_center'*v1_align_center;
        [U,~,V] = svd(H);
        R = V*U';
        if det(R)<0
            R(:,3)=-R(:,3);
        end
        T = v1_align_center'-R*v2_align_center';
        vtest=R*vtest'+repmat(T,1,size(vtest,1));
        vtest=vtest';
    end
    v_error = vtest-vground;
    groundtruthlist(i).v = vground;
    testlist(i).v = vtest;
    dist = sum(v_error.*v_error,2);
    modeli(i)=mean(sqrt(dist));
end
Erms = (mean(modeli));
end