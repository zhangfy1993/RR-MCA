% Build ridge regression model
classdef RR_model
    properties
        xcali;ycali;alpha;centeralization;meanx;meany;coef;
    end
    
    methods
        function obj = RR_model(xcali,ycali,alpha,centeralization)
            obj.xcali = xcali;
            obj.ycali = ycali;
            obj.alpha = alpha;
            obj.centeralization = centeralization;
            if centeralization
                [meanx,xx] = center(xcali);
                [meany,yy] = center(ycali);
                coef = RR(xx,yy,alpha);
                obj.meanx = meanx;
                obj.meany = meany;
            else
                coef = RR(xcali,ycali,alpha);
            end
            obj.coef = coef;
        end
        
        % Predict the y values
        function ytp = predict(obj,xtest)
            if obj.centeralization
                xxtest = xtest - repmat(obj.meanx,size(xtest,1),1);
                ytp = xxtest * obj.coef + obj.meany;
            else
                ytp = xtest * obj.coef;
            end
        end
        
        % Calculate the root mean square error of prediction
        function RMSE = get_rmse(obj,xtest,ytest)
            ytp = obj.predict(xtest);
            RMSE = norm(ytp-ytest)/sqrt(size(ytest,1));
        end
    end
end