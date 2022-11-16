function f3  = MSCN(img2)
if size(img2,3) ~= 1
img2 = rgb2gray(img2);
end
img2 = double(img2);

f3 = brisque_feature(img2);



function feat = brisque_feature(imdist)
scalenum = 2;
window = fspecial('gaussian',7,7/6);
window = window/sum(sum(window));
feat = [];
for itr_scale = 1:scalenum
mu            = filter2(window, imdist, 'same');
mu_sq         = mu.*mu;
sigma         = sqrt(abs(filter2(window, imdist.*imdist, 'same') - mu_sq));
structdis     = (imdist-mu)./(sigma+1);
[alpha,overallstd] = estimateggdparam(structdis(:));
feat               = [feat alpha overallstd^2]; 
imdist             = imresize(imdist,0.5);
end
%=======================================================
function [gamparam,sigma] = estimateggdparam(vec)
gam                = 0.2:0.001:10;
r_gam              = (gamma(1./gam).*gamma(3./gam))./((gamma(2./gam)).^2);
sigma_sq           = mean((vec).^2);
sigma              = sqrt(sigma_sq);
E                  = mean(abs(vec));
rho                = sigma_sq/E^2;
[~,array_position] = min(abs(rho - r_gam));
gamparam           = gam(array_position);