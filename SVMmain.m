
close all;
clear;
% tic
format compact;%空格紧凑
load ('record_pd1');
load ('img_data4');
load ('all_img.mat');

img_num = length(all_img);%

 for i=1:808
 img{i,1}=all_img{i,1};
 end
label=record_pd2';

load('feature.mat')
feature0=double(feature);



for n=1:1000
N=646;          
num=808;       
p=randperm(num);
a=p(1:N);    


[ kl1(n),mse1(n),chebyshev1(n),cor1(n),cosine1(n)]=SVMtrain0(img_num,a,feature0); 

end
kl11=sum(kl1)/n;
mse11=sum(mse1)/n;
chebyshev11=sum(chebyshev1)/n;
cor11=sum(cor1)/n;
cosine11=sum(cosine1)/n;
com(1,:)=[ kl11,mse11,chebyshev11,cor11,cosine11]
