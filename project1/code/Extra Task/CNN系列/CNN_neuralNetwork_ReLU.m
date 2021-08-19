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
nHidden = [100];

% Choose the kernel size
kernel_size = 5;

% weight decay
lambda = 1e-4;

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
w(1:kernel_size*kernel_size) = randn(kernel_size * kernel_size,1) * sqrt(2./(2+kernel_size*kernel_size));
w(1+kernel_size*kernel_size:nParams) = randn(nParams - kernel_size * kernel_size,1) * sqrt(2./(nParams - kernel_size)) ;

% Train with stochastic gradient
maxIter = 100000;
stepSize = 1e-3;
funObj = @(w,i)CNN_Loss_ReLU(w,X(i,:),yExpanded(i,:),kernel_size,nHidden,nLabels,lambda);
for iter = 1:maxIter
    if mod(iter-1,round(maxIter/20)) == 0
        yhat = CNN_Predict_ReLU(w,Xvalid,kernel_size,nHidden,nLabels);
        ytrain =  CNN_Predict_ReLU(w,X,kernel_size,nHidden,nLabels);
        Iter = [Iter,iter-1];
        Trainingerror =  [Trainingerror, sum(ytrain~=y)/t];
        Validationerror = [Validationerror, sum(yhat~=yvalid)/t];
        fprintf('Training iteration = %d, validation error = %f\n',iter-1,sum(yhat~=yvalid)/t);
    end
    
%     if mod(iter-1,50000) == 0
%         stepSize = stepSize/2;
%     end
    i = ceil(rand*n);
    [f,g] = funObj(w,i);s
    w = w - stepSize*g;
end

plot(Iter, [Trainingerror', Validationerror'])
legend('Training', 'Validation')
xlabel('Iter','FontSize',12);
ylabel('Error','FontSize',12);

% Evaluate test error
yhat = CNN_Predict_ReLU(w,Xtest,kernel_size,nHidden,nLabels);
fprintf('Test error with final model = %f\n',sum(yhat~=ytest)/t2);