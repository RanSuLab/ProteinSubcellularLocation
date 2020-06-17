function [wfeat1] = Haralick(readPath)
%HARALICK 此处显示有关此函数的摘要
%   此处显示详细说明
% dbtype = 'db1';
% readPath = 'C:\Users\ShifuBest\Downloads\code\feature\img2.jpg';
dbs = {'db1','db2','db3','db4','db5','db6','db7','db8','db9','db10'};
NLEVELS = 10;
GLEVELS = 31;
wfeat1 = [];
feattype = 'SLFs';

    for dbtype = dbs
        prot = imread(readPath);
%         A = uint8(round(GLEVELS*prot/max(prot(:))));
%         wfeat = ml_texture(A);
        offsets = [0 1; -1 1; -1 0; -1 -1];
        GLCM2 = graycomatrix(prot,'NumLevels',8,'Offset',offsets);
        stats = GLCM_Features1(GLCM2);
        wfeat = StructTomatrix(stats);
        wfeat = [mean(wfeat(1:13,[1 3]),2); mean(wfeat(1:13,[2 4]),2)]';
        [C,S] = wavedec2(prot,NLEVELS,char(dbtype));
        for k = 0 : NLEVELS-1
            [chd,cvd,cdd] = detcoef2('all',C,S,(NLEVELS-k));%提取高频部分
%             try 
                A = chd - min(chd(:));
%                 A = uint8(round(GLEVELS*A/max(A(:))));
%                 hfeat = ml_texture(A);
                GLCM2 = graycomatrix(A,'NumLevels',8,'Offset',offsets);
                stats = GLCM_Features1(GLCM2);
                hfeat = StructTomatrix(stats);
                hfeat = [mean(hfeat(1:13,[1 3]),2); mean(hfeat(1:13,[2 4]),2)]';
                A = cvd - min(cvd(:));
%                 A = uint8(round(GLEVELS*A/max(A(:))));
%                 vfeat = ml_texture(A);
                GLCM2 = graycomatrix(A,'NumLevels',8,'Offset',offsets);
                stats = GLCM_Features1(GLCM2);
                vfeat = StructTomatrix(stats);
                vfeat = [mean(vfeat(1:13,[1 3]),2); mean(vfeat(1:13,[2 4]),2)]';

                A = cdd - min(cdd(:));
%                 A = uint8(round(GLEVELS*A/max(A(:))));
%                 dfeat = ml_texture( A);
                GLCM2 = graycomatrix(A,'NumLevels',8,'Offset',offsets);
                stats = GLCM_Features1(GLCM2);
                dfeat = StructTomatrix(stats);
                dfeat = [mean(dfeat(1:13,[1 3]),2); mean(dfeat(1:13,[2 4]),2)]';

                wfeat = [wfeat hfeat vfeat dfeat ...
                    sqrt(sum(sum(chd.^2))) ...
                    sqrt(sum(sum(cvd.^2))) ...
                    sqrt(sum(sum(cdd.^2)))];
        
        end


        if strcmp(feattype,'SLFs')     
           wfeat1 = [wfeat1 wfeat];
        end
    end
%     disp(size(wfeat1));
end

