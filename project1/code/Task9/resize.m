function [new_img] = resize(img,area)

[m,n] = size(img);
    
% calculate corresponding position of each pixel in new image
center = [m,n]/2+0.5;
for i = 1:m
    for j = 1:n
        pos(i,j,:) = [(i-center(1))*area(1)/m+center(1),(j-center(2))*area(2)/n+center(2)];
    end
end

% get pixel value by linear interpolation
new_img = Bilinear_interpolation(img,pos);

end