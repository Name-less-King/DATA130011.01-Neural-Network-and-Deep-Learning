function Y = ReLU_bp(X)
    Y = zeros(size(X));
    Y(X>0) = 1;
end