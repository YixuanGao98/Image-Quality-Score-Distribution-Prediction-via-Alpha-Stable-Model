% close all;%关闭所有figure窗口
% clear;%清空变量
% 
% format compact;%空格紧凑
% load ('record_pd1');
% load ('img_data2');
% load ('all_img.mat');
% img_num = length(all_img);%获取图像总数量
% 
% for i=1:808
% img{i,1}=all_img{i,1};
% end
% 
% 
% img=img{100,1};
function feat = LBPgb(img)
if size(img,3)==3
    img_gray = rgb2gray(img);
else
    img_gray = img;
end
img_gray = uint8(img_gray);
mapDIST = LBP41(img_gray);

LBP=zeros(6,5);
[m,n]=size(mapDIST);
for i=1:m
    for j=1:n
        if mapDIST(i,j)==0
            LBP(1,1)= LBP(1,1)+1;
        end
         if mapDIST(i,j)==1
            LBP(1,2)= LBP(1,2)+1;
         end
         if mapDIST(i,j)==2
            LBP(1,3)= LBP(1,3)+1;
         end
         if mapDIST(i,j)==3
            LBP(1,4)= LBP(1,4)+1;
         end
         if mapDIST(i,j)==4
            LBP(1,5)= LBP(1,5)+1;
        end
    end
end
 LBP(1,:)=LBP(1,:)./(m*n);
sigma = [0.5 1 1.5 2 2.5];
for i = 1:length(sigma)
    win = fspecial('gaussian',3,sigma(i));
    imgPRI = imfilter(img,win);
    
    
    if size(imgPRI,3)==3
        imgPRI = rgb2gray(imgPRI);
    end
    imgPRI = uint8(imgPRI);
    
    mapPRI = LBP41(imgPRI);
    
    for k=1:m
        for j=1:n
            if mapPRI(k,j)==0
                LBP(i+1,1)=LBP(i+1,1)+1;
            end
            if mapPRI(k,j)==1
                LBP(i+1,2)= LBP(i+1,2)+1;
            end
            if mapPRI(k,j)==2
                LBP(i+1,3)= LBP(i+1,3)+1;
            end
            if mapPRI(k,j)==3
                LBP(i+1,4)= LBP(i+1,4)+1;
            end
            if mapPRI(k,j)==4
                LBP(i+1,5)= LBP(i+1,5)+1;
            end
        end
    end
    
      LBP(i+1,:)=LBP(i+1,:)./(m*n);
css(i)=sum(((LBP(i+1,:)-LBP(1,:)).^2)./(LBP(i+1,:)+LBP(1,:)));
feat(1,i) =css(i);
end

end




% function feat = LBPgb(img)
% % Input : (1) img: a RGB or gray scale image, and the dynamic range should be 0-255.
% % Output: (1) score: the quality score
% if size(img,3)==3
%     img = rgb2gray(img);
% end
% img = uint8(img);
% mapDIST = LBP41(img);
% 
% sigma = [0.5 1 1.5 2 2.5];
% for i = 1:length(sigma)
%     win = fspecial('gaussian',3,sigma(i));
%     imgPRI = imfilter(img,win);
%     
%     mapPRI = LBP41(imgPRI);
%     feat(1,i) = sum(sum(mapDIST.*mapPRI))/(sum(sum(mapPRI))+1);
% end
% 
% end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function LBP = LBP41(img)
% Calculate the LBP statistics using a neighbors number of 4 and a radius of 1
neighborNum = 4;
[M,N] = size(img);
% Coordinate offset of neighbors
offset = [0 1; -1 0; 0 -1; 1 0];
% Block size
bsize_M = 3;
bsize_N = 3;
% Starting coordinate
orig_m = 2;
orig_n = 2;
% d_m and d_n
d_m = M - bsize_M;
d_n = N - bsize_N;

% Center pixel matrix
Center = img(orig_m:orig_m+d_m,orig_n:orig_n+d_n);
% LBP matrix
LBP = zeros(d_m+1,d_n+1);

% Compute the LBP code matrix
D = cell(neighborNum,1);
for i = 1:neighborNum
    m = offset(i,1) + orig_m;
    n = offset(i,2) + orig_n;
    Neighbor = img(m:m+d_m,n:n+d_n);
    D{i} = Neighbor >= Center;
end

% Accumulate all neighbors
for i = 1:neighborNum
	LBP = LBP + D{i};
end

lbp_map = (LBP == 2) | (LBP == 3);

end

