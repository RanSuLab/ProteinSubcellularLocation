function [lbpfeat] = getLbp(readPath)
%GETLBP 此处显示有关此函数的摘要
spoints = [1, 0; 1, -1; 0, -1; -1, -1; -1, 0; -1, 1; 0, 1; 1, 1];
spoints2=[0,2;-1,2;-1,1;-2,1;-2,0;-2,-1;-1,-1;-1,-2;0,-2;1,-2;1,-1;2,-1;2,0;2,1;1,1;1,2];
prot = imread(readPath);
mapping = getmapping(8,'riu2');
lbpfeat1=LBP(prot,1,8,mapping,'h');
mapping = getmapping(16,'riu2');
lbpfeat2=LBP(prot,2,16,mapping,'h');
lbpfeat = [lbpfeat1 lbpfeat2];
end

