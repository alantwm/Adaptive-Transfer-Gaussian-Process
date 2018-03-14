classdef ATGP
% ATGP is a gaussian process-based supervised transfer learning algorithm 
% based on a paper by Cao et. al. This implementation utilizes the squared 
% exponential kernel function and the associated hyperparameters are optimized 
% using global search algorithm (CMAES).
%
% Described in detail in:
% Cao, Bin, Sinno Jialin Pan, Yu Zhang, Dit-Yan Yeung, and Qiang Yang. "Adaptive 
% Transfer Learning." In AAAI, vol. 2, no. 5, p. 7. 2010.
% 
% ATGP takes as inputs (x_target,y_target,x_source,y_source)
% Accepted inputs are shaped (n,d); n = # of instances, d = dimensions
% Accepted outputs are shaped (n,1)
% 
% Example Use:
% model = ATGP(x_target,y_target,x_source,y_source)
% yhat = model.predict(x_test)
    properties
        FeatureMap; % feature mapping matrix (size = TrFeatures x ScFeatures+1)
        hyp; % there are 5 hyperparams for SE Kernel Function ==> 
             % 1. char length (CovSE Kernel)
             % 2. covar func ampl.(CovSE Kernel) 
             % 3. source noise 
             % 4. target noise 
             % 5. inter-task similarity (lambda)             
        loglik; % marginal log likelihood function (to be maximized)
        n_source; % numer of source instances
        d_source; % number of input features in source data
        x_source; % source 'x' values normalized between 0 and 1
        y_source; % source 'y' values
        n_target; % number of target instances
        d_target; % number of input features in target data
        x_target; % target x values normalized between 0 and 1
        y_target; % target 'y' values
        L         % Intermediate values to speed up predictions
        alpha     % Intermediate values to speed up predictions
        bounds    % Bounds of hyp
    end
    methods 
        function obj = ATGP(x_target,y_target,x_source,y_source)
            obj.x_target = x_target;
            obj.y_target = y_target;
            obj.x_source = x_source;
            obj.y_source = y_source;
            [obj.n_source, obj.d_source] = size(x_source);
            [obj.n_target, obj.d_target] = size(x_target);                        
            obj.hyp = rand(1,5);
            obj = obj.train();  
            obj = obj.eval();
        end
        function K = covSE(obj,x,xp,sigma,l)
            % Generic Squared Exponential Covariance Matrix
            K=(sigma^2)*exp(-pdist2(x,xp).^2/(2*l^2));
        end
        function [K11,K12,K21,K22] = calc_K(obj)            
            K11 = obj.covSE(obj.x_source,obj.x_source,obj.hyp(2),obj.hyp(1));
            K11=K11+eye(obj.n_source,obj.n_source)*obj.hyp(3)^2;
            
            K22=obj.covSE(obj.x_target,obj.x_target,obj.hyp(2),obj.hyp(1));
            K22=K22+eye(obj.n_target,obj.n_target)*obj.hyp(4)^2;
            
            K12 = obj.covSE(obj.x_source,obj.x_target,obj.hyp(2),obj.hyp(1));
            
            K12 = obj.hyp(5)*K12;
            K21 = K12';
        end
        function obj = eval(obj)
            [K11,K12,K21,K22] = obj.calc_K();  
            L=chol(K11)';
            alpha = solve_chol(L',obj.y_source);
            v = L\K12;
            mut = K21*alpha;
            Ct = K22 - v'*v;

            try
                LL = chol(Ct);
            catch
                Ct = Ct+eye(size(Ct,1))*1e-8;
                LL = chol(Ct);
            end

            inv_piece = solve_chol(LL,(obj.y_target-mut));
            obj.loglik = -sum(log(diag(LL')))-0.5*(obj.y_target-mut)'*inv_piece;
        end
        function [mu,std] = predict(obj,x)
            %% Returns mean and std deviation of posterior distribution at x
            n=size(x,1);
            ks = zeros(n,obj.n_source+obj.n_target);
            kss = ones(n,1)*((obj.hyp(2)^2) + obj.hyp(4)^2);           
            
            ks_Sc = obj.hyp(5)*obj.covSE(obj.x_source,x,obj.hyp(2),obj.hyp(1));
            ks_Tr = obj.covSE(obj.x_target,x,obj.hyp(2),obj.hyp(1));
            ks = [ks_Sc;ks_Tr]';
            
            mu = ks*obj.alpha;
            v = obj.L\ks';
            var = kss - sum(v.*v)';
            
            % Fix for small var due to numerical error
            var(var<0 & var>-1e-8) = 0;
            std = var.^0.5;
        end
        
        function obj = fmodel(obj)
            [K11,K12,K21,K22] = obj.calc_K();            
            K= [K11,K12;K21,K22];
            try
                obj.L = chol(K)';
            catch
                K=K+eye(size(K,1))*1e-12;
                obj.L = chol(K)';
            end
            obj.alpha = solve_chol(obj.L',[obj.y_source;obj.y_target]);
        end
        function obj = train(obj)
            % Hyperparameter bounds in log space
            obj.bounds.lb = [ones(1,4)*log(1e-8) log(1e-8)];
            obj.bounds.ub = [ones(1,2)*log(5) ones(1,2)*log(1) log(1)];    
            
            %% CMAES
            opts = cmaes;
            opts.LBounds = ones(5,1)*0;
            opts.UBounds = ones(5,1)*1;
            opts.MaxIter = 500;
            opts.Restarts = 2;
            opts.SaveVariables = 'off';
                        
            [~, ~, ~, ~, ~, bestever] = cmaes('TGPfunc',rand(5,1),0.5,opts,obj);
            hyp = bestever.x;        
            hyp = hyp';
            
            % Updating model hyperparameters
            obj.hyp = exp(hyp.*(obj.bounds.ub-obj.bounds.lb)+obj.bounds.lb);
            obj = obj.fmodel();
        end
    end
end
function X = solve_chol(L, B)
    X = L\(L'\B);
end   