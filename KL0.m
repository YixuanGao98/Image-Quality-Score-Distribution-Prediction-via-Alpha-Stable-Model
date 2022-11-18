%KL,correlation,cosine

function [JSD ,mse,chebyshev,Chi_Square,cosine]=KL0(x,score,co)


nbins = 10;
data1 = score;  

    h = histogram(data1,'Normalization','pdf','BinLimits',[0,100],'BinWidth',100/nbins,...
                  'FaceColor',[0.27451 0.5098 0.70588],'EdgeColor','w','LineWidth',1);  %直方图
   p1= h.Values;
   X=0:100;
p2=pdf('stable',X,co(1),co(2),co(3),co(4));
for j=1:10
z(j,:)=p2((j-1)*10+1:(j-1)*10+10);
p3(j)=(sum(z(j,:))/10);
end
p2=p3;


for i=1:size(p1,2)
    if p1(i)==0
        p1(i)=0.0000000001;
    end
    if p2(i)==0
        p2(i)=0.0000000001;
    end
end

logQvect = log2((p2+p1)/2);
            kl = .5 * (sum(p1.*(log2(p1)-logQvect)) + ...
                sum(p2.*(log2(p2)-logQvect)));
JSD = JSDiv(p1,p2);
Y=[p1;p2];
spearman = abs(1-pdist(Y, 'Spearman'));
chebyshev = pdist(Y, 'chebychev');
err=p1-p2;
mse=sqrt(sum(err.^2)./nbins);%mse
cor=abs(1-pdist(Y,'correlation'));%Pearson 
cosine = 1- pdist(Y, 'cosine') ;
seuclidean = pdist(Y, 'seuclidean');
subMatrix = p1-p2;
subMatrix2 = subMatrix.^2;
addMatrix = p1+p2;
idxZero = find(addMatrix==0);
addMatrix(idxZero)=1;
DistMat = subMatrix2./addMatrix;
Chi_Square= sum(DistMat,2);  


end
   
