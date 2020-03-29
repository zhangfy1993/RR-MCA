# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

class RR_MCA(object):
    """ Build RR-MCA models """
    def __init__(self, X, y, M, yM, Xvali, yvali, weight='auto'):
        """ Initialization
        X: Spectra data of calibration samples
        y: The y values of calibration samples
        M: Spectra data of standard samples
        yM: The y values of standard samples
        Xvali: Spectra data of test samples
        yvali: The y values of test samples
        weight: The weight of standard samples
        """
        _, XX = center(X)
        meany, yy = center(y)
        meanM, MM = center(M)
        meanyM, yMM = center(yM)
        XXvali = Xvali - np.tile(meanM, (Xvali.shape[0], 1))
        yyvali = yvali - meanyM
        if weight == 'auto':
            weight = X.shape[0]
        spec = np.vstack((XX, weight*MM))
        target = np.hstack((yy, weight*yMM))
        self.X = X
        self.y = y
        self.M = M
        self.yM = yM
        self.Xvali = Xvali
        self.yvali = yvali
        self.weight = weight
        self.XX = XX
        self.yy = yy
        self.MM = MM
        self.yMM = yMM
        self.XXvali = XXvali
        self.yyvali = yyvali
        self.meany = meany
        self.meanyM = meanyM
        self.spec = spec
        self.target = target
               
    def optimize(self, coef, epsilon=0.8, alpha_min=1e-7, alpha_max=10, n_alpha=100):
        """ Optimize the RR parameter
        Inputs:
            coef: The regression coefficients of the original model
            epsilon (ε): The ratio of ||β*||2 over ||β||2, default: 0.8
            alpha_min: The minimal α value, default: 1e-7
            alpha_max: The maximum α value, default: 10
            n_alpha: The numbers of α values, default: 100
        Output:
            diff: The first columns: α values; The second column: abs(||β*||2-ε*||β||2) values
        """
        alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), n_alpha)
        diff = np.zeros(n_alpha)
        for i in range(n_alpha):
            coef_new = RR(self.spec, self.target, alphas[i])
            diff[i] = abs(np.linalg.norm(coef_new) - epsilon*np.linalg.norm(coef))
        plt.figure()
        plt.semilogx(alphas, diff, '-b.')
        plt.xlabel(u'α', fontsize = 12)
        plt.ylabel(u'||β$^{*}$||$_{2}$ - ε*||β||$_{2}$', fontsize = 12)
        plt.show()
        diff = np.vstack((alphas, diff))
        return diff.T
    
    def predict(self, alpha, plot=True):
        """ Predict and calculate the root mean square error """
        model = RR_model(self.spec, self.target, alpha, centeralization=False)
        
        mean_target = np.hstack((np.ones(self.X.shape[0])*self.meany, np.ones(self.M.shape[0])*self.meanyM))
        yc = np.hstack((self.y, self.yM))
        yt = self.yvali
        ycp = model.predict(np.vstack((self.XX, self.MM))) + mean_target
        ytp = model.predict(self.XXvali) + self.meanyM
        ymin = min(yc.min(), yt.min())
        ymax = max(yc.max(), yt.max())
        if plot:
            plt.figure()
            plt.plot(yc, ycp, 'ro')
            plt.plot(yt, ytp, 'b+')
            plt.plot([ymin, ymax], [ymin, ymax], 'k')
            plt.xlabel('Reference values', fontsize = 12)
            plt.ylabel('Predicted values', fontsize = 12)
            plt.show()
        rmse = model.get_rmse(self.XXvali, self.yyvali)
        return rmse
    
    def coef(self, alpha):
        """ Return the regression coefficients of the new model """
        model = RR_model(self.spec, self.target, alpha, centeralization=False)
        return model.coef

class RR_model(object):
    """ Build the ridge regression model """
    def __init__(self, xcali, ycali, alpha, centeralization=True):
        self.xcali = xcali
        self.ycali = ycali
        self.alpha = alpha
        self.centeralization = centeralization
        if centeralization:
            meanx, xx = center(xcali)
            meany, yy = center(ycali)
            coef = RR(xx, yy, alpha)
            self.meanx = meanx
            self.meany = meany
        else:
            coef = RR(xcali, ycali, alpha)
        self.coef = coef
        
    def predict(self, xtest):
        if self.centeralization:
            xxtest = xtest - np.tile(self.meanx, (xtest.shape[0], 1))
            ytp = np.dot(xxtest, self.coef) + self.meany
        else:
            ytp = np.dot(xtest, self.coef)
        return ytp
    
    def get_rmse(self, xtest, ytest):
        ytp = self.predict(xtest)
        RMSE = np.linalg.norm(ytp - ytest) / sqrt(ytest.shape[0])
        return RMSE

def RR(x, y, alpha):
    """ Return the regression coefficients of a RR model """
    I = np.eye(x.shape[1])
    coef = np.dot(np.linalg.inv(np.dot(x.T, x) + alpha*I), np.dot(x.T, y))
    return coef

def center(x):
    """ Center the data """
    if x.ndim == 1:
        meanx = np.mean(x)
        xx = x - meanx
    elif x.ndim == 2:
        meanx = np.mean(x, 0)
        xx = x - np.tile(meanx, (x.shape[0], 1))
    return meanx, xx
