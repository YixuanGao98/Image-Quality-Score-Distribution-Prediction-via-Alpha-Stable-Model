clear all

load ('all_img.mat');

img_num = length(all_img);%获取图像总数量


for i=1:length(all_img)
    % Read the image
img = all_img{i,1};

feature(i,1:4) =MSCN(img);
feature(i,5:24) = my_feature(img);

feature(i,:)= Impulsenoise_feature(img);

feature(i,:)=MultiplicativeNoise(img);
end