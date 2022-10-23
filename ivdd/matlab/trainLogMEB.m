
function [alpha,r,costHistory,violatorsHistory] = trainLogMEB(K,C,sparseInit,toll,maxIt,verbosity,kP)

useMinusProbs = 0;
%% init
% default 1e-4
toll2 =1e-4;
% default 1e-4
toll22 = 1e-4;
% use .1 with linear kernel. use 0.01 in case of problems
lambda1 = .01;
% r self consistence stabilizer, default 0.1. use 0.01 in case of problems
lambda2 = .01;

kk = diag(K);

n = length(kk);

probs = zeros(n,1);
grad = zeros(n+1,1);
% n*C/2 is a bound
% n*C/4 is the agnostic value      
%barR = (n*C/4);
barR = 1;
%barR = 10;
alpha = [];

if (sparseInit==1)
    alpha = zeros(n,1);           
else
    alpha = 1/n*ones(n,1);  
end
it = 0;
varPart = zeros(n,1);
% first computation
Ka = K*alpha;
constPart = alpha'*Ka;
for i=1:n
     varPart(i)=-2*K(i,:)*alpha+kk(i);
end

dist = constPart+varPart;
minusprobs = zeros(n,1);
% init probabilities

if (useMinusProbs==1)
    for i=1:n            
        minusprobs(i)=OneMinusProbKer(dist(i)-barR,kP);
    end
else
    for i=1:n            
        probs(i) = probKer(dist(i)-barR,kP);
    end    
end

% gradient init
grad(2:n+1)=0; 

if (useMinusProbs==1)
    for i=1:n                    
        p1 = minusprobs(i);
        gg = Ka-K(:,i);
        grad(2:n+1) = grad(2:n+1)+p1.*gg*kP;    
    end
else
    for i=1:n                    
        pp = probs(i);
        gg = Ka-K(:,i);
        grad(2:n+1) = grad(2:n+1)+(1-pp).*gg*kP;
    end
end

        
grad(2:n+1)=2*C*grad(2:n+1);

if (useMinusProbs==1)
    grad(1) = -C*sum(minusprobs)*kP+2*barR;
else
    grad(1) = -C*sum(1-probs)*kP+2*barR;    
end

violators = zeros(n,1);

violatorsHistory = [];
costHistory = [];

    zz=1;
    barRold = 1e20;
    while(zz<100)
        acc = 0;             
        if (useMinusProbs==1)
            % update probabilities
            for i=1:n            
                minusprobs(i) = OneMinusProbKer(dist(i)-barR,kP);
                acc = acc+minusprobs(i);
            end
        else
            % update probabilities
            for i=1:n            
                probs(i) = probKer(dist(i)-barR,kP);
                acc = acc+(1-probs(i));
            end            
        end
        barR = lambda2*kP*acc*C/2+(1-lambda2)*barR;
        zz = zz+1;        
        if (abs(barR-barRold)<toll22)
            break
        end        
        barRold = barR;
    end

   
%if (verbosity>=2)
%    costHistory = [doCost(probs,barR,C)];
%    violatorsHistory = [sum(alpha~=0)];
%end
    
innerIt = 0;
%% loop
while(1)

    %innerIt = innerIt+1;
    if (verbosity>=1)
    if (useMinusProbs==1)
        cost = doCostm(minusprobs,barR,C);
    else
        cost = doCost(probs,barR,C);
    end
        fprintf('\n it %d Cost %.6f max grad %f barR %f',it,cost,max(abs(grad)),barR);
        %fflush(stdout);
    end

    worstIndex = 0;
    worstViolator = 0;
    
    % worst optimality conditions violator
    for k=1:n                
        if (abs(grad(k+1))>worstViolator)
            worstViolator = abs(grad(k+1));
            worstIndex = k;
        end        
    end 
       
    if (verbosity>=3)
        fprintf('\nWorst violator %d...',worstIndex);
    end
    
    % cache worst index
    cache = K(worstIndex,:);    
    violators(worstIndex)=1;
    
    % optimize recursively the worst violator 
    alphaOld = alpha(worstIndex);
    alphaOpt = alphaOld;
    alphaOptOld = alphaOpt;

    innerIt = 0;
    % alpha loop
    while(innerIt<100)    
        innerIt = innerIt+1;
        ker = cache*alpha;
        ker = ker - alpha(worstIndex)*cache(worstIndex);
      
        num = 0;
        den = 0;
        
        if (useMinusProbs==1)
            for k=1:n
                num = num - (minusprobs(k))*(ker-cache(k));
                den = den + (minusprobs(k));
            end
        else
            for k=1:n
                %fprintf('\n strange term %f',ker-cache(k));
                num = num - (1-probs(k))*(ker-cache(k));
                den = den + (1-probs(k));
            end          
        end
               
        % non smooth iterations (dangerous!)
        %alphaOpt = num/(den*cache(worstIndex));
        
        % opt alpha        
        alphaOpt = lambda1*num/(den*cache(worstIndex))+(1-lambda1)*alphaOld;
        
        % modified line. 
        %alphaOpt = lambda1*num/(den*cache(worstIndex))+(1-lambda1)*alphaOptOld;
        
        % update alpha
        alpha(worstIndex)=alphaOpt;

        % closed form update of ||phi-a|| when changing one alpha                        
        % quadratic part of the update (global)        
                
         dist = dist+cache(worstIndex)*(alphaOpt^2-alphaOptOld^2)+2*ker*(alphaOpt-alphaOptOld);
    
         if (useMinusProbs==1)
             for k=1:n
                % linpart of dist is pattern specific (local)
                dist(k)=dist(k)-2*cache(k)*(alphaOpt-alphaOptOld);
                % update 1-probabilities too
                minusprobs(k) = OneMinusProbKer(dist(k)-barR,kP);
             end
         else
             for k=1:n
                % linpart of dist is pattern specific (local)
                dist(k)=dist(k)-2*cache(k)*(alphaOpt-alphaOptOld);
                % update probabilities too
                probs(k) = probKer(dist(k)-barR,kP);
             end        
         end
        
        if (verbosity>=3)
            fprintf('*');
        end
        
        %fprintf('\n\t Violator %d %f ',worstIndex,alphaOpt);
        %fprintf('\n %.3f %.3f n %.20f d %.20f p %.20f ',alphaOptOld,alphaOpt,num,(den*cache(worstIndex)),probs(worstIndex));
        %if (innerIt>10)
        %    probs
        %    pause();
        %end
        
        %fprintf('\n test %f',abs(alphaOptOld-alphaOpt));
        % check if it is necessary to stop inner loop
        if (abs(alphaOptOld-alphaOpt)<toll2)
            %fprintf('\n\t Finished in %d iterations',innerIt);
            %pause();
            break;
        end

        alphaOptOld = alphaOpt;
    end
    %

    % efficient update of Ka vector    
    Ka = Ka+K(:,worstIndex).*(alphaOpt-alphaOld);    
        
    % gradient update    
    grad(2:n+1)=0;
        
    if (useMinusProbs==1)        
        for i=1:n                    
            p1 = minusprobs(i);
            gg = Ka-K(:,i);
            grad(2:n+1) = grad(2:n+1)+p1.*gg*kP;
        end   
    else
        for i=1:n                    
            pp = probs(i);
            gg = Ka-K(:,i);
            grad(2:n+1) = grad(2:n+1)+(1-pp).*gg*kP;
        end 
    end    
    grad(2:n+1)=2*C*grad(2:n+1);
        
    % we have still to check it for optimality
    if (useMinusProbs==0)
        grad(1) = -C*sum(1-probs)*kP+2*barR;
    else
        grad(1) = -C*sum(minusprobs)*kP+2*barR;
    end
    
    if (verbosity>=2)
        costHistory = [costHistory cost];
        violatorsHistory = [violatorsHistory sum(alpha~=0)];
    end
    
    % recursive solution on barR    
    zz=1;
    barRold = 1e20;
    while(zz<100)
        acc = 0;             
        if (useMinusProbs==1)
            % update probabilities
            for i=1:n            
                minusprobs(i) = OneMinusProbKer(dist(i)-barR,kP);
                acc = acc+minusprobs(i);
            end
        else
            % update probabilities
            for i=1:n            
                probs(i) = probKer(dist(i)-barR,kP);
                acc = acc+(1-probs(i));
            end            
        end
        barR = lambda2*kP*acc*C/2+(1-lambda2)*barR;
        zz = zz+1;        
        if (abs(barR-barRold)<toll22)
            break
        end        
        barRold = barR;
    end
        
    % stop when max gradient component is under tollerance    
    % check only when the gradient is fully updated
    if (max(abs(grad))<toll)
        break;
    end
        
    it = it+1;
    if (it>=maxIt)
        break;
    end
            
    barRold = barR;    
end

if (useMinusProbs==1)
        cost = doCostm(minusprobs,barR,C);
else
        cost = doCost(probs,barR,C);
end
    
fprintf('\n it %d Cost %.6f max grad %f barR %f',it,cost,max(abs(grad)),barR);
r = sqrt(barR);