function [ltpfeat] = getLTP(readPath)
%GETLTP �˴���ʾ�йش˺�����ժҪ
%   �˴���ʾ��ϸ˵��
prot = imread(readPath);

[ ltp_upper, ltp_lower ] = LTP(prot,5);
% disp(size(ltp_upper));
mapping = getmapping(16,'riu2');

[ ltp_upper1, ltp_lower1 ] = LTP(prot,2);
ltp_upperLBP=LBP(ltp_upper,2,16,mapping,'h');
ltp_lowerLBP=LBP(ltp_lower,2,16,mapping,'h');
ltp_upper1LBP=LBP(ltp_upper1,2,16,mapping,'h');
ltp_lower1LBP=LBP(ltp_lower1,2,16,mapping,'h');
ltpfeat = [ltp_upperLBP ltp_lowerLBP ltp_upper1LBP ltp_lower1LBP];
end

