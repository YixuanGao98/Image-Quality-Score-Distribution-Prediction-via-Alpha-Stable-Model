
function cof=SVMtrain(img_num,a,feature)
load ('img_data4');
img_data2=img_data4(:,1:779);
load ('record_pd1');
record_pd1=record_pd1(1:779,:);
fid = fopen('ref_ids.txt');
allref_ids=textscan(fid, '%f');

allref_ids=allref_ids{1,1};
m=1;
l=1;
flag=zeros(1,img_num);
for i=1:length(allref_ids)
    if ismember(allref_ids(i),a)
        flag(1,i)=1;
    end
end

for i=1:img_num
    if flag(1,i) ==1
        img_data_train(:,m)=img_data2(:,i);
    train_data(m,:)=feature(i,:);
      train_label(m,:)=record_pd1(i,:);
    m=m+1;
    elseif flag(1,i) ==0
        img_data_test(:,l)=img_data2(:,i);
    test_data(l,:)=feature(i,:); 
    test_label(l,:)=record_pd1(i,:); 
    l=l+1;
    end
end
%
[train_data,ps1] = mapminmax(train_data');
train_data = train_data';
test_data = mapminmax('apply',test_data',ps1);
test_data = test_data';
%%
%alpha 
train_label_alpha=train_label(:,1);%
test_label_alpha=test_label(:,1);
[train_label_alpha,ps2] = mapminmax(train_label_alpha');
train_label_alpha = train_label_alpha';
test_label_alpha = mapminmax('apply',test_label_alpha',ps2);
test_label_alpha = test_label_alpha';
%beta
train_label_beta=train_label(:,2);%
test_label_beta=test_label(:,2);
 %gam
train_label_gam=train_label(:,3);%
test_label_gam=test_label(:,3);
%% delt
train_label_delt=train_label(:,4);%
test_label_delt=test_label(:,4);
%
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


model= svmtrain(train_label_delt,train_data, '-s 3 -t 2 -r 0 -c 0.5  -p 0.1 -g 0.5 -q');
[predicted_label_delt] = svmpredict(test_label_delt, test_data, model,'-b 0 ');
predicted_label_delt = mapminmax('reverse',predicted_label_delt,ps5);
test_label_delt= mapminmax('reverse',test_label_delt,ps5);

%predicted paramaters
cof=[predicted_label_alpha,predicted_label_beta,predicted_label_gam,predicted_label_delt];
end