function [new_img] = rotation(img,theta)

[m,n] = size(img);
    
% calculate corresponding position of each pixel in new image
for i = 1:m
    for j = 1:n
        x = j-n/2-0.5;
        y = i-m/2-0.5;
        pos(i,j,:) = [-x*sin(theta)+y*cos(theta)+m/2+0.5,x*cos(theta)+y*sin(theta)+m/2+0.5];
    end
end

% get pixel value by linear interpolation
new_img = Bilinear_interpolation(img,pos);

end