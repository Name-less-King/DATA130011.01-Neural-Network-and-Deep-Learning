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


% used for plot
Trainingerror = [];
Validationerror = [];
Iter = [];

% Count number of parameters and initialize weights 'w'
nParams = d*nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;
w = randn(nParams,1) * sqrt(2./nParams);


% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;
funObj = @(w,i)Fast_MLPclassificationLoss_ReLU(w,X(i,:),yExpanded(i,:),nHidden,nLabels);
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/200)) == 0
        yhat = MLPclassificationPredict_ReLU(w,Xvalid,nHidden,nLabels);
        ytrain =  MLPclassificationPredict_ReLU(w,X,nHidden,nLabels);
        Iter = [Iter,iter-1];
        Trainingerror =  [Trainingerror, sum(ytrain~=y)/t];
        Validationerror = [Validationerror, sum(yhat~=yvalid)/t];
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
    end
    
   
    i = ceil(rand*n);
    [f,g] = funObj(w,i);
    w = w - stepSize*g;
    
end

plot(Iter, [Trainingerror', Validationerror'])
legend('Training', 'Validation')
xlabel('Iter','FontSize',12);
ylabel('Error','FontSize',12);

% Evaluate test error
yhat = MLPclassificationPredict_ReLU(w,Xtest,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);