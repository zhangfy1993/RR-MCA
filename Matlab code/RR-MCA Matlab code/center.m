function [meanx,xx] = center(x)
% Center the data
meanx = mean(x);
xx = x - repmat(meanx,size(x,1),1);
end