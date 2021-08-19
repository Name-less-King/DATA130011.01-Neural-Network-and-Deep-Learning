function [model] = leastSquaresBasis(x,y,degree)

% Construct Basis
Xpoly = polyBasis(x,degree);

% Solve least squares problem (assumes that Xpoly'*Xpoly is invertible)
w = (Xpoly'*Xpoly)\Xpoly'*y;

model.w = w;
model.degree = degree;
model.predict = @predict;

end

function [yhat] = predict(model,Xtest)
Xpoly = polyBasis(Xtest,model.degree);
yhat = Xpoly*model.w;
end

function [Xpoly] = polyBasis(x,m)
n = length(x);
Xpoly = zeros(n,m+1);
for i = 0:m
    Xpoly(:,i+1) = x.^i;
end
end