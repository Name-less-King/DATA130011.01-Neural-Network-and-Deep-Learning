addpath('../')

load digits.mat
[n,d] = size(X);
nLabels = max(y);
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

% Standardize columns and add bias
[X,mu,sigma] = standardizeCols(X);
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xtest = standardizeCols(Xtest,mu,sigma);
% X = [ones(n,1) X];
% d = d + 1;
% 
% % Make sure to apply the same transformation to the validation/test data
% Xvalid = standardizeCols(Xvalid,mu,sigma);
% Xvalid = [ones(t,1) Xvalid];
% Xtest = standardizeCols(Xtest,mu,sigma);
% Xtest = [ones(t2,1) Xtest];

% Choose network structure
nHidden = [200];

% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;
w = randn(nParams,1) * sqrt(2./nParams);

% used for weight decay
lambda = 1e-3;

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;
funObj = @(w,i)Fast_MLPclassificationLoss_ReLU_w(w,X(i,:),yExpanded(i,:),nHidden,nLabels,lambda);
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/200)) == 0
        yhat = MLPclassificationPredict_ReLU(w,Xvalid,nHidden,nLabels);
        ytrain = MLPclassificationPredict_ReLU(w,X,nHidden,nLabels);
        fprintf('Training iteration = %d, train error = %f, validation error = %f\n',iter-1,sum(ytrain~=y)/t,sum(yhat~=yvalid)/t);
    end
    
%     if mod(iter,100000)==0
%         stepSize = stepSize/2;
%     end
    i = ceil(rand*n);
    [f,g] = funObj(w,i);
    w = w - stepSize*g;
    
end

% Evaluate test error
yhat = MLPclassificationPredict_ReLU(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);