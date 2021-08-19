function [y] = linearInd2Binary(ind,nLabels)

n = length(ind);

y = -ones(n,nLabels);

for i = 1:n
    y(i,ind(i)) = 1;
end