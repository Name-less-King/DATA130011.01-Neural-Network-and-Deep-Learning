function [f,g] = SoftmaxLoss_drop(w,X,y,nHidden,nLabels,p)

[nInstances,nVars] = size(X);

% Form Weights
inputWeights = reshape(w(1:nVars*nHidden(1)),nVars,nHidden(1));
offset = nVars*nHidden(1);
for h = 2:length(nHidden)
  hiddenWeights{h-1} = reshape(w(offset+1:offset+(nHidden(h-1)+1)*nHidden(h)),nHidden(h-1)+1,nHidden(h));
  offset = offset+(nHidden(h-1)+1)*nHidden(h);
end
hiddenWeights{length(nHidden)} = w(offset+1:offset+(nHidden(end)+1) * nLabels);
hiddenWeights{length(nHidden)} = reshape(hiddenWeights{length(nHidden)},nHidden(end)+1,nLabels);

f = 0;

% generate the mask for dropout
for h = 1:length(nHidden)
    Mask{h} = rand(1, nHidden(h))<p;
end

% Compute Output
for i = 1:nInstances
    
    ip{1} = X(i,:)*inputWeights;
    fp{1} = [1,Mask{1} .* tanh(ip{1})];
    for h = 2:length(nHidden)
        ip{h} = fp{h-1}*hiddenWeights{h-1};
        fp{h} = [1,Mask{h} .* tanh(ip{h})];
    end
    yhat = Softmax(fp{end}*hiddenWeights{end});
    
    f = f + (- log(yhat(y(i))));
    
    if nargout > 1
        err = yhat;
        err(y(i)) = err(y(i)) - 1;
        for h = length(nHidden):-1:1
            gHidden{h} = fp{h}' * err;
            err = Mask{h} .* sech(ip{h}).^2 .* (err * hiddenWeights{h}(2:nHidden(h)+1,:)');
        end
        gInput = X(i,:)' * err;
    end

end

% Put Gradient into vector
if nargout > 1
    g = zeros(size(w));
    g(1:nVars*nHidden(1)) = gInput(:);
    offset = nVars*nHidden(1);
    for h = 2:length(nHidden)
        g(offset+1:offset+(nHidden(h-1)+1)*nHidden(h)) = gHidden{h-1};
        offset = offset+(nHidden(h-1)+1)*nHidden(h);
    end
    g(offset+1:offset+(nHidden(end)+1)*nLabels) = gHidden{end}(:);
end
