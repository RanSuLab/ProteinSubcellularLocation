function processAll()
%UNTITLED �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
imgPath='E:/hpaData/Multilabel/';
imgDataDir  = dir(imgPath);             % ���������ļ�
for i = 1:length(imgDataDir)
%     if(isequal(imgDataDir(i).name,'.')||... % ȥ��ϵͳ�Դ����������ļ���
%        isequal(imgDataDir(i).name,'..')||...
%        ~imgDataDir(i).isdir)                % ȥ�������в����ļ��е�
%            continue;
%     end
    imgDir = dir([imgPath imgDataDir(i).name '/*.jpg']); 
    for j =1:length(imgDir)  
        % ��������ͼƬ
        disp('����');
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

