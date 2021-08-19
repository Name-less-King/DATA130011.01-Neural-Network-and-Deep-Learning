addpath('../')

load digits.mat
[n,d] = size(X);
nLabels = max(y);
% yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
X = [ones(n,1) X];
d = d + 1;

% Make sure to apply the same transformation to the validation/test data
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xvalid = [ones(t,1) Xvalid];
Xtest = standardizeCols(Xtest,mu,sigma);
Xtest = [ones(t2,1) Xtest];

% Choose network structure
nHidden = [10];

%dropout rate
p = 0.5;

% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+(nHidden(h-1)+1) * nHidden(h);
end
nParams = nParams+(nHidden(end)+1) * nLabels;
w = randn(nParams,1);

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-2;
funObj = @(w,i)SoftmaxLoss_drop(w,X(i,:),y(i,:),nHidden,nLabels,p);
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = SoftmaxPredict_drop(w,Xvalid,nHidden,nLabels,p);
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
    end
    
    i = ceil(rand*n);
    [f,g] = funObj(w,i);
    w = w - stepSize*g;
end

% Evaluate test error
yhat = SoftmaxPredict_drop(w,Xtest,nHidden,nLabels,p);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);