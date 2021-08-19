function [new_img] = translation(img,trans)

[m,n] = size(img);
    
% calculate corresponding position of each pixel in new image
for i = 1:m
    for j = 1:n
        pos(i,j,:) = [i-trans(1),j-trans(2)];
    end
end

% get pixel value by linear interpolation
new_img = Bilinear_interpolation(img,pos);

end