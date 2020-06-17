function [wfeat] = StructTomatrix(stats)
%STRUCTTO 此处显示有关此函数的摘要
%   此处显示详细说明
wfeat(1,:) =  cat(1,stats.contr);
        wfeat(2,:) =  cat(1,stats.corrp);
        wfeat(3,:) =  cat(1,stats.energ);
        wfeat(4,:) =  cat(1,stats.entro);
        wfeat(5,:) =  cat(1,stats.homop);
        wfeat(6,:) =  cat(1,stats.sosvh);
        wfeat(7,:) =  cat(1,stats.savgh);
        wfeat(8,:) =  cat(1,stats.svarh);
        wfeat(9,:) =  cat(1,stats.senth);
        wfeat(10,:) =  cat(1,stats.dvarh);
        wfeat(11,:) =  cat(1,stats.denth);
        wfeat(12,:) =  cat(1,stats.inf1h);
        wfeat(13,:) =  cat(1,stats.inf2h);
end

