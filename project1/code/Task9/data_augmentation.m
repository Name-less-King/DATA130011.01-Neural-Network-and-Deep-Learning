addpath('../')
% set how much images should be added
new_image = 10000;

load digits.mat;
N = round(new_image/size(X,1));

% the sample id, it start with size(X,1)+1 if we want to add more samples
id = size(X,1)+1;

for i = 1:size(X,1)
    for j = 1:floor(N)
        
        % reshape into 16*16 image
        img = reshape(X(i,:),16,16)/255;
        
        % randomly decide one kind of the three type of data augmentation
        type = floor(rand()*3);
        switch type
            case 0
                trans = 4*rand(1,2)-2;          
                img = translation(img, trans);
            case 1
                theta = 0.6*rand()-0.3;                 
                img = rotation(img,theta);
            case 2
                area = floor(14+5*rand(1,2));         
                img = resize(img,area);
        end
        
        % add into trainnig set
        X(id,:) = floor(255*img(:));
        y(id,:) = y(i,:);
        id = id + 1;
    end
end

% save the new data 
save(['augmentation_digits.mat'],'Xtest','ytest','X','y','Xvalid','yvalid');

