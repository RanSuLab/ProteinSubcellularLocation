function getAllFeature()
%GETALLFEATURE 此处显示有关此函数的摘要
%   此处显示详细说明
path_test = 'E:/myProgram/python/deepLearning/model/Data_python/AlexNet/data_fold_crop_aug_CLAHE/1_fold/test1';
path_train = 'E:/myProgram/python/deepLearning/model/Data_python/AlexNet/data_fold_crop_aug_CLAHE/1_fold/train1';
path_all = 'E:/hpaData/del_notGood/all';
path = path_all;
%     for p = 2:2
%         if p==2            % 遍历所有文件
%             path = path_train;
%         end
        imgDataDir = dir(path);   
%         for i = 1:length(imgDataDir)%1174 690 
         for i = 1:length(imgDataDir)%1174 690 
            imgDir = dir([path imgDataDir(i).name '/*.jpg']); 
%             for j =   4508:length(imgDir) 
             for j =  877:length(imgDir) 
                  readPath = [path  '/' imgDir(j).name];
                  S = regexp(imgDir(j).name, '_', 'split');
                  S{3}(end:end)=[];
                  label = S{3};
                  haralick = Haralick(readPath);
                  haralick(isnan(haralick))=0;
                  lbp = getLbp(readPath);
                  ltp = getLTP(readPath);
                  lqp = getLQP(readPath);
                  feature = [lbp ltp lqp haralick];
                  disp(j);
                  location = ['A',int2str(j)];
                  location1 = ['B',int2str(j)];
                  xlswrite('C:\Users\ShifuBest\Downloads\code\feature\feature2.xlsx', feature, 'Sheet1',location1);
                  xlswrite('C:\Users\ShifuBest\Downloads\code\feature\feature2.xlsx', {label}, 'Sheet1',location);
            end
        end
%     end
end

