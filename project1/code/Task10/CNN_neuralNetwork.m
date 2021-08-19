addpath('../')

load digits.mat
[n,d] = size(X);
nLabels = max(y); 
yExpanded = linearInd2Binary(y,nLabels);
t = size(Xvalid,1);
t2 = size(Xtest,1);

[X,mu,sigma] = standardizeCols(X);
Xvalid = standardizeCols(Xvalid,mu,sigma);
Xtest = standardizeCols(Xtest,mu,sigma);

% Choose network structure
nHidden = [10];

% Choose the kernel size
kernel_size = 5;

% used for plot
Trainingerror = [];
Validationerror = [];
Iter = [];

% Count number of parameters and initialize weights 'w'
nParams = kernel_size * kernel_size + 144 * nHidden(1);
for h = 2:length(nHidden)
    nParams = nParams+nHidden(h-1)*nHidden(h);
end
nParams = nParams+nHidden(end)*nLabels;
w = randn(nParams,1);

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;
funObj = @(w,i)CNN_Loss(w,X(i,:),yExpanded(i,:),kernel_size,nHidden,nLabels);
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = CNN_Predict(w,Xvalid,kernel_size,nHidden,nLabels);
        ytrain =  CNN_Predict(w,X,kernel_size,nHidden,nLabels);
        Iter = [Iter,iter-1];
        Trainingerror =  [Trainingerror, sum(ytrain~=y)/t];
        Validationerror = [Validationerror, sum(yhat~=yvalid)/t];
        fprintf('Training iteration = %d, validation error = %f, traing error = %f\n',iter-1,sum(yhat~=yvalid)/t,sum(ytrain~=y)/t);
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
yhat = CNN_Predict(w,Xtest,kernel_size,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);