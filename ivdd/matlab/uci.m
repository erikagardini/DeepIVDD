clc;
clear all;

filename = 'breast';

for num=0:0
    for seed=0:0
        kP = 25;
        err1V = [];
        err2V = [];
        sparsityV = [];
        nitV = [];
        entropyV = [];
        % 1 original, 2 simple (always non sparse)
        algo = 1;

        %% settings
        C = 0.1700; %0 = 0.16, 1=0.2, 2=0.17, 3 = 0.17
        %C = 80.;
        %ktype = 'linear';
        ktype = 'rbf';
        sparseInit = 1;
        %toll =  1e-2;
        toll =  1e-1;
        maxIt = 3000;
        doCheck = 1;
        refClass = 1;
        tgSize = .5;
        normalize = 1;
        verbosity = 2;
        randn('state',0);
        rand('state',1);

        if (doCheck==1)
            verbosity = 2;
        end
        
        Xtg = load(strcat('./uci/', filename, '_seed',int2str(seed), '_train.txt'));
        Xval1 = load(strcat('./uci/', filename,'_seed',int2str(seed), '_val1.txt'));
        Xval2 = load(strcat('./uci/', filename,'_seed',int2str(seed), '_val2.txt'));
        
        n = size(Xtg,1); %"normali" per training
        nv1 = size(Xval1,1); %sottoninsieme di "normali" per test
        nv2 = size(Xval2,1); %anomali

        %% kernel
        C_hat = C/n;
        CC = C;
        C_old = -1
        fprintf('\nComputing kernel...');
        tic
        D = calcdist(Xtg,Xtg);
        %sigma = 8*max(D(:))/sqrt(2*n);
        sigma = max(D(:))/log(n);  
        %sigma = 1;
        fprintf('\nsigma is %f',sigma);
        K = calckernel(ktype,sigma,Xtg);
        Ktime = toc;


        in = 0.0;
        incr = 0.01;
        decr = 0;
        while in < 80.0 | in > 90.0

            %% training
            fprintf('\nTraining..');
            tic

            if (algo==1)
                [alpha,r,costHistory,violatorsHistory] = trainLogMEB(K,C_hat,sparseInit,toll,maxIt,verbosity,kP);
            elseif (algo==2)
                [alpha,r,costHistory,violatorsHistory] = trainLogMEBsimple2(K,C_hat,sparseInit,toll,maxIt,verbosity,kP);
            end

            tgTime = toc;

            % sparsity
            cc = 0;
            for i=1:n    
                if (alpha(i)==0)        
                    cc = cc+1;       
                end
            end
            cc = cc/n*100;

            constPart = alpha'*K*alpha;
            ftg = zeros(n,1);
            ftg2 = zeros(n,1);
            for i=1:n
               varPart=-2*K(i,:)*alpha+K(i,i);        
               dist  = (constPart+varPart);
               ftg(i) = (r*r-dist);
            end

            ptg = 1./(1+exp(-kP*ftg));

            in = 0;
            for i=1:n
                if (ptg(i)>=.5)
                    in = in+1;
                end
            end
            in= in/n*100;

            fprintf('\n Training time %f kernel time %f',tgTime,Ktime);
            fprintf('\n Inside %.2f %% Sparsity %.2f %%\n',in,cc);
            fprintf('\n C %f',C);
            
            if in < 80.0
                %incr=incr/10;
                C_old = C;
                C=C+incr;
                decr = 1;
            end
            if in > 90.0
                C_old = C;
                C = C-incr;
                if decr == 1
                    incr=incr/10;
                    C = C+incr;
                    decr = 0;
                end
            end
            C_hat = C/n;

        end
        
        %% check

        if (doCheck==1)
             figure;
             plot((1-violatorsHistory/n)*100);
             grid on;
             %title('Sparsity');
  
             hh = figure;
             %plot(costHistory,'.-b');
             %hold on;
             %plot((1-violatorsHistory/n)*100,'-.r');
             t1 = linspace(1,length(costHistory),length(costHistory));
             t2 = linspace(1,length(violatorsHistory),length(violatorsHistory));
             [hAx,hLine1,hLine2] = plotyy(t1,costHistory,t2,(1-violatorsHistory/n)*100);
             set(hLine1,'LineWidth',3);
             set(hLine2,'LineWidth',3);
             set(hLine1,'LineStyle','-');
             set(hLine2,'LineStyle','--');   
             %set(hLine1,'Marker','.');
             %set(hLine2,'Marker','.');
             grid on;
             saveas(hh, strcat('./probs_uci/loss_',filename,'_seed',int2str(seed),'.png'));

             close;
             close;
             %save('toy.mat');
             dlmwrite(strcat('./probs_uci/', filename, '_seed', int2str(seed), '_alpha.txt'),alpha)

             %axis tight;
             %title('Cost function');
        end

        %% test

        % these patterns should stay in 
        % f<0 means out thus it is an error
        Kt = calckernel(ktype,sigma,Xtg,Xtg);
        Kself = calckernelself(ktype,Xtg);
        [f_train, p_train]= testLogMeb(K,Kt,Kself,alpha,r,kP);

        Kt = calckernel(ktype,sigma,Xtg,Xval1);
        Kself = calckernelself(ktype,Xval1);
        [f1 p1]= testLogMeb(K,Kt,Kself,alpha,r,kP);
        nt1 = length(f1);
        th = 0.5;
        % out patterns
        %err1 = sum(f1<0)/nt1*100;
        err1 = sum(p1<th)/nt1*100;
        err1num = sum(f1<0);
        % these patterns should stay out 
        % f>0 means in thus it is an error    
        Kt = calckernel(ktype,sigma,Xtg,Xval2);
        Kself = calckernelself(ktype,Xval2);
        [f2 p2] = testLogMeb(K,Kt,Kself,alpha,r,kP);
        nt2 = length(f2);
        % in patterns
        %err2 = sum(f2>0)/nt2*100 ;
        err2 = sum(p2>th)/nt2*100 ;
        err2num = sum(f2>0);
        fprintf(' False inner %.2f %% False outliers %.2f %%\n\n',err1,err2);
        fprintf('\nerr %f',(sum(f1<0)+sum(f2>0))/(nt1+nt2)*100);
        fprintf('\nBER KLO %f\n',.5*(err1+err2));

        % figure
        %hist([p1; p2],25);
        bins = hist([p1; p2],25);
        bins = bins./(length(p1)+length(p2));
        H = sum(-(bins.*(log2(bins))));
        entropyV = [entropyV H];

        err1V = [err1V err1];
        err2V = [err2V err2];
        sparsityV = [sparsityV cc]; 

        fprintf('\nMean e1 %.1f +- %.1f\nmean e2 %.1f +- %.1f\nmean ber %.1f\nmean sparsity %.1f\n',mean(err1V),std(err1V),mean(err2V),...
        std(err2V),.5*(mean(err1V)+mean(err2V)),mean(sparsityV));
        fprintf('\nMean entropy in bits %.1f\n',mean(entropyV));

        probability = cat(1, p1, p2);
        y_val1 = ones(nv1,1);
        y_val2 = zeros(nv2,1);
        actual = cat(1, y_val1, y_val2);
            
        fileID = fopen(strcat('./results_uci/', filename, '_seed', int2str(seed),'.txt'),'w');
        fprintf(fileID,'C %f',C_old);
        fprintf(fileID,'\nInside %.2f',in);
        fprintf(fileID,'\nBer %.1f',.5*(err1+err2));
        
        [X,Y,T,AUC]=perfcurve(actual,probability,1);
        fprintf(fileID, '\nAUC %f\n',AUC);
        fprintf('\nAUC2 %f\n',AUC);
        
        dlmwrite(strcat('./probs_uci/', filename, '_seed', int2str(seed), '_p_test.txt'), probability, 'precision','%.20f')
        
        f = figure;
        h2 = histogram(probability,10)
        set(h2,'facecolor','blue','facealpha',1)

        saveas(f, strcat('./probs_uci/' , filename,'_seed',int2str(seed),'_test.png'))
        close;
        
        dlmwrite(strcat('./probs_uci/', filename, '_seed', int2str(seed), '_p_train.txt'), p_train, 'precision','%.20f')

        f = figure;
        h2 = histogram(p_train,10)
        set(h2,'facecolor','blue', 'facealpha', 1)
        saveas(f, strcat('./probs_uci/' , filename,'_seed',int2str(seed), '_train.png'))
        
        close;
        
        fclose(fileID);
    end
end
