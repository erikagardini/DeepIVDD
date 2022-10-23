
function kaux = calckernelself(ktype,X)

n = size(X,1);
kaux = ones(n,1);
if (strcmpi(ktype,'linear'))
   for i=1:n
    kaux(i) = X(i,:)*X(i,:)';
   end
elseif (strcmpi(ktype,'rbf'))
   return;
end