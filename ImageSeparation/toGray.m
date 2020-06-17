function toGray()
%TOGRAY 此处显示有关此函数的摘要
%   此处显示详细说明
img=imadjust(dcm(i).img,[0.5;0.55],[0;1]); %调整灰度范围2   
img=double(img);              %将灰度级映射到0~2553    
low=min(min(img));
high=max(max(img));
maxgray=high-low;
%计算窗宽6      
rate=256/maxgray;
img=dcm(i).img*rate;
img=img+abs(min(min(img)));    %加窗9  
img=uint8(img);           %转化为8位的位图数据格式



end

