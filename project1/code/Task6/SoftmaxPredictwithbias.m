function [y] = SoftmaxPredictwithbias(w,X,nHidden,nLabels)
[nInstances,nVars] = size(X);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+(nHidden(h-1)+1)*nHidden(h)),nHidden(h-1)+1,nHidden(h));
  offset = offset+(nHidden(h-1)+1) * nHidden(h);
end
outputWeights = w(offset+1:offset+(nHidden(end)+1)*nLabels);
outputWeights = reshape(outputWeights, nHidden(end) +1 ,nLabels);

% Compute Output
for i = 1:nInstances
    ip{1} = X(i,:)*inputWeights;
    fp{1} = [1,tanh(ip{1})];
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = [1,tanh(ip{h})];
    end
    y(i,:) = Softmax(fp{end}*outputWeights);
end
[v,y] = max(y,[],2);      
% y: the largest number position in each row