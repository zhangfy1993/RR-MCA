# -*- coding: utf-8 -*-
from scipy.io import loadmat
from RR_MCA import RR_MCA, RR_model

# Load data
data = loadmat('tablet.mat')
x1_cali = data['x1_cali']
x1_test = data['x1_test']
x2_cali = data['x2_cali']
x2_test = data['x2_test']
xt1 = data['xt1']
xt2 = data['xt2']
y_cali = data['y_cali']
y_test = data['y_test']
yt = data['yt']
wavelength = data['wavelength']
y_cali = y_cali[:, 0]
y_test = y_test[:, 0]
yt = yt[:, 0]
wavelength = wavelength[0, :]

# Original model built on the primary instrument
model = RR_model(x1_cali, y_cali, 6.58e-4)

# RR-MCA model
f = RR_MCA(x1_cali, y_cali, xt2, yt, x2_test, y_test)

# Optimize model parameter
f.optimize(model.coef, epsilon=0.8, alpha_min=1e-9, alpha_max=1000, n_alpha=100)

# Update and predict
f.predict(1.52e-3)