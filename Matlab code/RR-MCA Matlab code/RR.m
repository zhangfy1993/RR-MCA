function coef = RR(x,y,alpha)
% Return the regression coefficients of a ridge regression (RR) model
% x: spectra data
% y: y values
% alpha: RR model parameter
I = eye(size(x,2));
coef = (x'*x+alpha*I)\x'*y;
end