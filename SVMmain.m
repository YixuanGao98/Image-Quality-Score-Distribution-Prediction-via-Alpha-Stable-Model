
close all;
clear;
tic
load ('record_pd1');
record_pd1=record_pd1(1:779,:);
load ('all_img.mat');
img_num = 779;
for i=1:img_num
img{i,1}=all_img{i,1};
end
load('feature.mat')
for n=1:1000
N=23;          
num=29;     
p=randperm(num);
a=p(1:N);    
Pre_cof=SVMtrain(img_num,a,feature);
end