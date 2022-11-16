
function [img] = impulsenoise(f,ND)

    img = rgb2gray(f);
    ND = ND;
    NT = 1;
 
Narr = rand(size(img));
if isempty(NT) || NT == 0
    img(Narr<ND/2) = 0;
    img((Narr>=ND/2)&(Narr<ND)) = 255;
elseif NT == 1
    N = Narr;
    N(N>=ND)=0;
    N1 = N;
    N1 = N1(N1>0);
    Imn=min(N1(:));
    Imx=max(N1(:));
    N=(((N-Imn).*(255-0))./(Imx-Imn));
    img(Narr<ND) = N(Narr<ND);
else
    disp('Invalid selection');
end