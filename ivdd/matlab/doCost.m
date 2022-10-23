function cost = doCost(pp,barR,C)

n = length(pp);
% compute cost
cost = 0;
for i=1:n
    cost = cost-log(pp(i));
end
   
cost = barR^2 + C*cost;
