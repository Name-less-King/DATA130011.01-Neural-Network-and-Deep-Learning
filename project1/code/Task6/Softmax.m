function [yhat] = Softmax(y)
    yhat = exp(y)./sum(exp(y));
end