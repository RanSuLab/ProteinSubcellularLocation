function H = findH( V, W)
% H = FINDH( V, W) linearly unmixes a set of N m-dimensional samples (Nxm matrix)
%   inputs: V, the original samples  V是原始样本
%           W the color basis matrix, m-dimensioanl, r=rank (mxr)
%           W是颜色基础矩阵m维 r阶
% 
%   outputs: H, the unmixed weight channels
%输出H是未混合的权重通道

H = single(V)*pinv(single(W));
H = H - min(H(:));
H = H / max(H(:))*255;
H = uint8(round(H));%舍入到最接近的整数; 

