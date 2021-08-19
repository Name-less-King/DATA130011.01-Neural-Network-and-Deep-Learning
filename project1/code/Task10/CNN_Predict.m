function [y] = CNN_Predict(w,X,kernel_size,nHidden,nLabels)
nInstances = size(X,1);
nVars = 144;
% Form Weights
kernelWeights = reshape(w(1:kernel_size * kernel_size),kernel_size,kernel_size);
offset = kernel_size * kernel_size;
inputWeights = reshape(w(offset+1:offset + nVars * nHidden(1)),nVars,nHidden(1));
offset = offset + nVars * nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+nHidden(h-1)*nHidden(h)),nHidden(h-1),nHidden(h));
  offset = offset+nHidden(h-1)*nHidden(h);
end
outputWeights = w(offset+1:offset+nHidden(end)*nLabels);
outputWeights = reshape(outputWeights,nHidden(end),nLabels);

% Compute Output
for i = 1:nInstances
    x = reshape(X(i,:),16,16)/255;
    
    Conv = conv2(x,kernelWeights,'valid');
    Conv = reshape(Conv,1,size(Conv,1)*size(Conv,2));
    
    ip{1} = Conv * inputWeights;
    fp{1} = tanh(ip{1});
    for h = 2:length(nHidden)
        ip{h} = fp{h-1} * hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    y(i,:) = fp{end} * outputWeights;
end
[v,y] = max(y,[],2);
%y = binary2LinearInd(y);
