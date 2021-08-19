function [new_img] = Bilinear_interpolation(img,pos)
% input:    img: the original image                                 (size M*N)
%           pos: the coresponding position in the original image    (size M*N*2)
% output:   new_img: the new image                                  (size M*N)

% the size of image
[M,N] = size(img);
new_img = zeros(M,N);

% add zero padding to image
pimg = zeros(M+2,N+2);
pimg(2:M+1,2:N+1) = img;

for i = 1:M
    for j = 1:N
        % the corresponding postion in original image
        x = pos(i,j,1);
        y = pos(i,j,2);
        
        % if the postion is in the scope of padding image
        if x>0 && y>0 && x<M+1 && y<N+1
            intx = floor(x);
            inty = floor(y);
            
            % linear interpolation weights
            s = x-intx;
            t = y-inty;
            
            % calculate new pixel value
            tmp =(1-s)*(1-t)*pimg(intx+1,inty+1) + (1-s)*t*pimg(intx+1,inty+2) + s*(1-t)*pimg(intx+2,inty+1) + s*t*pimg(intx+2,inty+2);
            new_img(i,j) = floor(255*tmp)/255;
        end
    end
end

end



