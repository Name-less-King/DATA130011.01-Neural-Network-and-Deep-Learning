function [f,g] = CNN_Loss(w,X,y,kernel_size,nHidden,nLabels)
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
hiddenWeights{length(nHidden)} = w(offset+1:offset+nHidden(end)*nLabels);
hiddenWeights{length(nHidden)} = reshape(hiddenWeights{length(nHidden)},nHidden(end),nLabels);

f = 0;

% Compute Output
for i = 1:nInstances
    x = reshape(X(1,:),16,16)/255;
    
    Conv = conv2(x,kernelWeights,'valid');
    Conv_reshape = reshape(Conv,1,size(Conv,1)*size(Conv,2));
    
    % first fully connected layer
    ip{1} = Conv_reshape * inputWeights;
    fp{1} = tanh(ip{1});
    
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = tanh(ip{h});
    end
    
    yhat = fp{end} * hiddenWeights{end};
    
    relativeErr = yhat-y(i,:);
    f = f + sum(relativeErr.^2);
    
    if nargout > 1
        err = 2 * relativeErr;
        for h = length(nHidden):-1:1
            gHidden{h} = fp{h}'* err;
            err = sech(ip{h}).^2 .* (err * hiddenWeights{h}');
        end
        
        % weights gradient between last convolutional layer with
        % first fully connected layer
        gInput = Conv_reshape'* err;
        err = reshape(err*inputWeights',size(Conv));
        
        
        % weights gradient of convolutional kernels
        x_reverse = reshape(X(i,end:-1:1), 16, 16)/255;
        gConv = conv2(x_reverse, err, 'valid');
        
    end
    
    % Put Gradient into vector
    if nargout > 1
        g = zeros(size(w));
        g(1:kernel_size*kernel_size) = gConv(:);
        offset = kernel_size * kernel_size;
        g(offset+1:offset + nVars*nHidden(1)) = gInput(:);
        offset = offset + nVars*nHidden(1);
        for h = 2:length(nHidden)
            g(offset+1:offset+nHidden(h-1)*nHidden(h)) = gHidden{h-1};
            offset = offset+nHidden(h-1)*nHidden(h);
        end
        g(offset+1:offset+nHidden(end)*nLabels) = gHidden{end}(:);
    end
end
