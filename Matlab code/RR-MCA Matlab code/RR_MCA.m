classdef RR_MCA
    properties
        X;y;M;yM;Xvali;yvali;weight;XX;yy;MM;yMM;XXvali;yyvali;meany;meanyM;spec;target;
    end
    
    methods
        function obj = RR_MCA(X,y,M,yM,Xvali,yvali)
            [~,XX] = center(X);
            [meany,yy] = center(y);
            [meanM,MM] = center(M);
            [meanyM,yMM] = center(yM);
            XXvali = Xvali - repmat(meanM,size(Xvali,1),1);
            yyvali = yvali - meanyM;
            weight = size(X,1);
            spec = [XX;weight*MM];
            target = [yy;weight*yMM];
            obj.X = X;
            obj.y = y;
            obj.M = M;
            obj.yM = yM;
            obj.Xvali = Xvali;
            obj.yvali = yvali;
            obj.weight = weight;
            obj.XX = XX;
            obj.yy = yy;
            obj.MM = MM;
            obj.yMM = yMM;
            obj.XXvali = XXvali;
            obj.yyvali = yyvali;
            obj.meany = meany;
            obj.meanyM = meanyM;
            obj.spec = spec;
            obj.target = target;
        end
        
        % Optimize the ridge regression model parameter
        function diff = optimize(obj,coef,epsilon,alpha_min,alpha_max,n_alpha)
            alphas = logspace(log10(alpha_min),log10(alpha_max),n_alpha)';
            diff = zeros(n_alpha,1);
            for i=1:n_alpha
                coef_new = RR(obj.spec,obj.target,alphas(i));
                diff(i) = abs(norm(coef_new) - epsilon*norm(coef));
            end
            figure();
            semilogx(alphas, diff, '-b.')
            xlim([alpha_min,alpha_max])
            xlabel('¦Á','FontSize',12)
            ylabel('||¦Â^{*}||_{2}-¦Å*||¦Â||_{2}','FontSize',12)
            diff = [alphas diff];
        end
        
        % Predict and calculate the root mean square error
        function RMSE = predict(obj,alpha)
            model = RR_model(obj.spec,obj.target,alpha,false);
            mean_target = [ones(size(obj.X,1),1)*obj.meany;ones(size(obj.M,1),1)*obj.meanyM];
            yc = [obj.y;obj.yM];
            yt = obj.yvali;
            ycp = model.predict([obj.XX;obj.MM])+mean_target;
            ytp = model.predict(obj.XXvali)+obj.meanyM;
            ymin = min([yc;yt]);
            ymax = max([yc;yt]);
            figure();
            hold on
            plot(yc, ycp, 'ro')
            plot(yt, ytp, 'b+')
            plot([ymin,ymax], [ymin,ymax], 'k')
            xlabel('Reference values', 'FontSize', 12)
            ylabel('Predicted values', 'FontSize', 12)
            RMSE = model.get_rmse(obj.XXvali,obj.yyvali);
        end
        
        % Return the regression coefficients of the new model
        function coef = get_coef(obj,alpha)
            model = RR_model(obj.spec,obj.target,alpha,false);
            coef = model.coef;
        end
    end
end

