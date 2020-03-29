load tablet

% Original model built on the primary instrument
model = RR_model(x1_cali,y_cali,6.58e-4,true);

% RR-MCA model
f = RR_MCA(x1_cali,y_cali,xt2,yt,x2_test,y_test);

% Optimize model parameter
f.optimize(model.coef,0.8,1e-9,1000, 100);

% Update and predict
RMSE = f.predict(1.52e-3)


