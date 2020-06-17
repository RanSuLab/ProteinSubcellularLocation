function [lqpfeat] = getLQP(readPath)
prot = imread(readPath);
[ lqp_upper11, lqp_lower11,lqp_upper12, lqp_lower12 ] = LQP(prot,2,5);
[ lqp_upper21, lqp_lower21,lqp_upper22, lqp_lower22 ] = LQP(prot,5,8);
[ lqp_upper31, lqp_lower31,lqp_upper32, lqp_lower32 ] = LQP(prot,2,8);
[ lqp_upper41, lqp_lower41,lqp_upper42, lqp_lower42 ] = LQP(prot,5,11);
mapping = getmapping(16,'riu2');
lqp_upperLBP11=LBP(lqp_upper11,2,16,mapping,'h');
lqp_lowerLBP11=LBP(lqp_lower11,2,16,mapping,'h');
lqp_upperLBP12=LBP(lqp_upper12,2,16,mapping,'h');
lqp_lowerLBP12=LBP(lqp_lower12,2,16,mapping,'h');
lqp_upperLBP21=LBP(lqp_upper21,2,16,mapping,'h');
lqp_lowerLBP21=LBP(lqp_lower21,2,16,mapping,'h');
lqp_upperLBP22=LBP(lqp_upper22,2,16,mapping,'h');
lqp_lowerLBP22=LBP(lqp_lower22,2,16,mapping,'h');
lqp_upperLBP31=LBP(lqp_upper31,2,16,mapping,'h');
lqp_lowerLBP31=LBP(lqp_lower31,2,16,mapping,'h');
lqp_upperLBP32=LBP(lqp_upper32,2,16,mapping,'h');
lqp_lowerLBP32=LBP(lqp_lower32,2,16,mapping,'h');
lqp_upperLBP41=LBP(lqp_upper41,2,16,mapping,'h');
lqp_lowerLBP41=LBP(lqp_lower41,2,16,mapping,'h');
lqp_upperLBP42=LBP(lqp_upper42,2,16,mapping,'h');
lqp_lowerLBP42=LBP(lqp_lower42,2,16,mapping,'h');
lqpfeat1 = [lqp_upperLBP11 lqp_lowerLBP11 lqp_upperLBP12 lqp_lowerLBP12];
lqpfeat2 = [lqp_upperLBP21 lqp_lowerLBP21 lqp_upperLBP22 lqp_lowerLBP22];
lqpfeat3 = [lqp_upperLBP31 lqp_lowerLBP31 lqp_upperLBP32 lqp_lowerLBP32];
lqpfeat4 = [lqp_upperLBP41 lqp_lowerLBP41 lqp_upperLBP42 lqp_lowerLBP42];
lqpfeat = [lqpfeat1 lqpfeat2 lqpfeat3 lqpfeat4];
end

