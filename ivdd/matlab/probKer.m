function p = probKer(f,k)

if (length(k)==1)
p = 1/(1+exp(k*f));
else
p = 1/(1+exp(k(1)*f+k(2))); 
end