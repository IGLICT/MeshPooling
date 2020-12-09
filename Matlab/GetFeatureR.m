function [ ] = GetFeatureR( srcfolder, number )
cmdline = ['.\GenFeature.exe 16 ',srcfolder,' ', num2str(number)];
dos(cmdline);
tarfvt = [srcfolder,'\fv_r.mat'];
movefile('E:\SIGA2014\workspace\fv.mat',tarfvt);
%movefile('F:\SIGA2014\workspace\fv.mat',tarfvt);
end
