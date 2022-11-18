
function [ kl0,mse0,chebyshev0,cor0,cosine0]=SVMtrain0(img_num,a,feature0)
load ('img_data2');
load ('record_pd2');
img_data2=img_data2;
record_pd1=record_pd2;
m=1;
l=1;
flag=zeros(1,img_num);%代表100张图片是否在a中，是1，否则0
for i=1:size(a,2)
    flag(1,a(1,i))=1;
end
% s=sum(flag);

for i=1:img_num
    if flag(1,i) ==1
        img_data_train(:,m)=img_data2(:,i);%原始评分数据也分成训练集，测试集
    train_data(m,:)=feature0(i,:);%特征
      train_label(m,:)=record_pd1(i,:);%label,四个参数值
    m=m+1;
    elseif flag(1,i) ==0
        img_data_test(:,l)=img_data2(:,i);
    test_data(l,:)=feature0(i,:); 
    test_label(l,:)=record_pd1(i,:); 
    l=l+1;
    end
end






%输入样本归一化
[train_data,ps1] = mapminmax(train_data');
train_data = train_data';
test_data = mapminmax('apply',test_data',ps1);
test_data = test_data';



%%
%alpha 

train_label_alpha=train_label(:,1);%测试集变量
test_label_alpha=test_label(:,1);
%输出样本归一化
[train_label_alpha,ps2] = mapminmax(train_label_alpha');
train_label_alpha = train_label_alpha';
test_label_alpha = mapminmax('apply',test_label_alpha',ps2);
test_label_alpha = test_label_alpha';

%%
% 
%beta

train_label_beta=train_label(:,2);%测试集变量
test_label_beta=test_label(:,2);



% %输出样本归一化
% [train_label_beta,ps3] = mapminmax(train_label_beta');
% train_label_beta = train_label_beta';
% test_label_beta = mapminmax('apply',test_label_beta',ps3);
% test_label_beta = test_label_beta';

 %%
 %gam
train_label_gam=train_label(:,3);%测试集变量
test_label_gam=test_label(:,3);


%% delt
train_label_delt=train_label(:,4);%测试集变量
test_label_delt=test_label(:,4);
%输出样本归一化
[train_label_delt,ps5] = mapminmax(train_label_delt');
train_label_delt = train_label_delt';
test_label_delt = mapminmax('apply',test_label_delt',ps5);
test_label_delt = test_label_delt';

% 
model= svmtrain(train_label_alpha,train_data, '-s 3 -t 2 -r 0 -c 0.5 -n 0.1 -p 0.001 -g 0.5 -q');
[predicted_label_alpha] = svmpredict(test_label_alpha, test_data, model,'-b 0 ');
predicted_label_alpha = mapminmax('reverse',predicted_label_alpha,ps2);
predicted_label_alpha = mapminmax(predicted_label_alpha',0,2);%0-2
predicted_label_alpha=predicted_label_alpha';
test_label_alpha= mapminmax('reverse',test_label_alpha,ps2);

model= svmtrain(train_label_beta,train_data, '-s 3 -t 2 -r 0 -c 0.5 -n 0.1 -p 0.01 -g 0.5 -q');
[predicted_label_beta] = svmpredict(test_label_beta, test_data, model,'-b 0 ');
predicted_label_beta= mapminmax(predicted_label_beta');
predicted_label_beta=predicted_label_beta';

model= svmtrain(train_label_gam,train_data, '-s 3 -t 2 -r 0 -c 0.5 -n 0.01 -p 0.1 -g 0.5 -q');
[predicted_label_gam] = svmpredict(test_label_gam, test_data, model,'-b 0 ');


model= svmtrain(train_label_delt,train_data, '-s 3 -t 2 -r 0 -c 0.5  -p 0.1 -g 0.5 -q');%CORNIA 3,1 其余3,2
[predicted_label_delt] = svmpredict(test_label_delt, test_data, model,'-b 0 ');
predicted_label_delt = mapminmax('reverse',predicted_label_delt,ps5);
test_label_delt= mapminmax('reverse',test_label_delt,ps5);

%预测的参数
cof=[predicted_label_alpha,predicted_label_beta,predicted_label_gam,predicted_label_delt];

% 
% x=[5,15,25,35,45,55,65,75,85,95];
x=[];
for i=1:162
   [ kl(i),mse(i),chebyshev(i),cor(i),cosine(i)]=KL0(x,img_data_test(:,i),cof(i,:));
end
kl0=nansum(kl)/162;%越小越好
mse0=nansum(mse)/162;%越小越好
chebyshev0=nansum(chebyshev)/162;%越小越好
cor0=nansum(cor)/162;%越大越好
cosine0=nansum(cosine)/162;%越大越好
% spearman0=sum(spearman)/162;
end