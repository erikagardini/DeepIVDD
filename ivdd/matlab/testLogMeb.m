
function [f,p] = testLogMeb(K,Kt,Kself,alpha,r,kP)
% Gives vector f and p. f>0 means ok, f<0 means error.
% p is probability vector.

barR = r*r;
nv = size(Kt,1);
f = zeros(nv,1);
constPart = alpha'*K*alpha;
for i=1:nv
    kaux = Kself(i);
    %-2*Kt(i,:)*alpha
    varPart=-2*Kt(i,:)*alpha+kaux;
    dist  = constPart+varPart;
    %f(i) = (barR-dist)/(dist);
    f(i) = (barR-dist);
end    

p = 1./(1+exp(-kP*f));