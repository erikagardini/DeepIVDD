
function D = calcdist(X1,X2)

dim=size(X1,2);
n1=size(X1,1);

if nargin==2
    n2=size(X2,1);
end

if nargin==2
    D = sqrt((repmat(sum(X1.*X1,2)',n2,1) + repmat(sum(X2.*X2,2),1,n1) ...
            - 2*X2*X1')); 
else    
    P=sum(X1.*X1,2);
    D = sqrt(repmat(P',n1,1) + repmat(P,1,n1)- 2*X1*X1');
end
