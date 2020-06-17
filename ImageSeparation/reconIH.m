function [J,ma] = reconIH( I,H,IDX,sc)
% J = RECONIH( I, H) reconstruct an NMF unmixed image
%   Takes as input I, the image to be unmixed, and H, the sorted pixel-weight channels.
%   Returns J, the unmixed image. Should be called after findWH.m
%J = RECONIH（I，H）重建NMF未混合图像
%作为输入I，要混合的图像，以及H，已排序的像素 - 权重通道。
%返回J，未混合的图像。 应该在findWH.m之后调用
if ~exist( 'IDX','var')
    IDX = [];
end
if ~exist( 'sc','var')
    sc = 0;
end

ma = [];

s = size(I);

if ~length(IDX)
    [u i IDX] = unique( 255-reshape( I, [s(1)*s(2) s(3)]), 'rows');%输出互不相同的行
end

J = H(IDX,:);
if sc
    ma = zeros(size(H,2),1);%size(H,2)矩阵列数
    for i=1:size(H,2)
        [c b] = imhist(J(:,i));
        [a ind] = max(c);
        J(:,i) = J(:,i)-b(ind);
        ma(i) = max(J(:,i));
    end
    J = (255/max(J(:)))*J;
end
J = reshape( J, [s(1) s(2) size(H,2)]);% s(1)=292 s(2)=294

if size(H,2)<3
    J(:,:,3) = 0;
end

