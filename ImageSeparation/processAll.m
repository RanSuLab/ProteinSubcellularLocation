function processAll()
%UNTITLED 此处显示有关此函数的摘要
%   此处显示详细说明
imgPath='../Multilabel/';
imgDataDir  = dir(imgPath);             % 遍历所有文件
for i = 1:length(imgDataDir)
%     if(isequal(imgDataDir(i).name,'.')||... % 去除系统自带的两个隐文件夹
%        isequal(imgDataDir(i).name,'..')||...
%        ~imgDataDir(i).isdir)                % 去除遍历中不是文件夹的
%            continue;
%     end
    imgDir = dir([imgPath imgDataDir(i).name '/*.jpg']); 
    for j =1:length(imgDir)  
        % 遍历所有图片
        disp('第张');
        disp(j);
        readPath = [imgPath imgDataDir(i).name '/' imgDir(j).name];%E:/hpaData/hpaAll../UBTF_cerebral+cortex_Nucleolus3.jpg

        imgWritePath1=strrep(readPath,'Multilabel','ProcessMultiLabel/DNA');
        imgWritePath2=strrep(readPath,'Multilabel','ProcessMultiLabel/protein');
        imgWritePath3=strrep(readPath,'Multilabel','ProcessMultiLabel/unmix_composition');
        testProcessImage(readPath,imgWritePath1,'DNA');
        testProcessImage(readPath,imgWritePath2,'protein');
        testProcessImage(readPath,imgWritePath3,'unmix_composition');
    end
end

end

