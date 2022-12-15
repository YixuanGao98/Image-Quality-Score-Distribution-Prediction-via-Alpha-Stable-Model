%选取合适特征值，在减去BLI56基础上,再减去一，DIVI的第三个特征
close all;%关闭所有figure窗口
clear;%清空变量
% tic
format compact;%空格紧凑
load ('record_pd2');
record_pd1=record_pd2;
load ('img_data2');
load ('all_img.mat');

img_num = length(all_img);%获取图像总数量

% for i=1:808
% img{i,1}=all_img{i,1};
% end
label=record_pd2';

% 
% f1=load('BMPRI_feature.mat');
% % f2=load('BPRI_feature.mat');
% % f3=load('BLI_feature.mat');
% % f4=load('DIVINE_feature.mat');
% % f5=load('BRI_feature.mat');
% % f6=load('nferm_feature.mat');
% % f7=load('CNN_feature.mat');
% % f8=load('VGG_feature.mat');
% % f9=load('CORNIA_feature.mat');
% % f10=load('NIQE_feature.mat');
% % % f12=load('newBMPRI_feature3.mat');%只修改了分母和参数
% % f11=load('newBMPRI_feature5.mat');%以卡方为准则的相似性
% % f12=load('BIQI_feature.mat');%与中间质量的距离，以cosin为准则，
% % 
% % 
% feature0=double(f1.feature);


% 
%%%%%my method
f6=load('nferm_feature.mat');
f11=load('newBMPRI_feature5.mat');
f1=load ('QAM16feature.mat');
f2=load ('spatial_corfeature2.mat');
f3=load ('ImpilseNoisefeature.mat');
f4=load ('MultiplicativeNoisefeature');
f5=load ('raylNoisefeature');
feature0(:,1:4)=double (f6.feature(:,1:4));
feature0(:,5:24)=f11.feature;

%0.0070    0.0061    0.0143    0.0173    0.9075//0.0007    0.0004    0.0011    0.0016    0.0085
feature0(:,25:29)=f4.MultiplicativeNoisefeature;%0.0072    0.0062    0.0148    0.0178    0.9065// 0.0024    0.0016    0.0050    0.0051    0.0085
feature0(:,30:34)=f3.ImpulseNoisefeature;

% 随机80-20，1000次
for n=1:10
N=646;           %需要抽取的图片的数量,646训练,162测试
num=808;       %图片的总数量
p=randperm(num);%随机生成1~num个随机整数
a=p(1:N);     %取p的前N个数


[ kl1(n),mse1(n),chebyshev1(n),cor1(n),cosine1(n)]=SVMtrain0(img_num,a,feature0); 

%  
 n

end
%平均值
kl11=sum(kl1)/n;
mse11=sum(mse1)/n;
chebyshev11=sum(chebyshev1)/n;
cor11=sum(cor1)/n;
cosine11=sum(cosine1)/n;
com(1,:)=[ kl11,mse11,chebyshev11,cor11,cosine11]
mse1=mse1';
cosine1=cosine1';
%%标准差
sdkl11=std(kl1);
sdmse11=std(mse1);
sdchebyshev11=std(chebyshev1);
sdcor11=std(cor1);
sdcosine11=std(cosine1);
sdcom(1,:)=[ sdkl11,sdmse11,sdchebyshev11,sdcor11,sdcosine11]
% kl22=sum(kl2)/n;
% mse22=sum(mse2)/n;
% chebyshev22=sum(chebyshev2)/n;
% cor22=sum(cor2)/n;
% cosine22=sum(cosine2)/n;
% com(2,:)=[ kl22,mse22,chebyshev22,cor22,cosine22];
% % 
% kl33=sum(kl3)/n;
% mse33=sum(mse3)/n;
% chebyshev33=sum(chebyshev3)/n;
% cor33=sum(cor3)/n;
% cosine33=sum(cosine3)/n;
% com(3,:)=[ kl33,mse33,chebyshev33,cor33,cosine33];
% 
% kl44=sum(kl4)/n;
% mse44=sum(mse4)/n;
% chebyshev44=sum(chebyshev4)/n;
% cor44=sum(cor4)/n;
% cosine44=sum(cosine4)/n;
% com(4,:)=[ kl44,mse44,chebyshev44,cor44,cosine44];
% com