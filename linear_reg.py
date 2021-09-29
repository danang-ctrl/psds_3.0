#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 08:19:46 2021

@author: dn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class lin_reg:
    def __init__(self,x_train=[], y_train=[]):
        self.x_train = x_train
        self.y_train = y_train
    def predict(self):
        x_bar = np.mean(self.x_train)
        y_bar = np.mean(self.y_train)
        n = len(self.x_train)
        for i in range(n):
            b_1 = np.sum((self.x_train[i]-x_bar)*(self.y_train[i]-y_bar))/np.sum((self.x_train[i]-x_bar)**2)
        b_0 = y_bar - b_1*x_bar
        y_predict = b_0 + b_1*self.x_train
        return y_predict
        
# Test use in real data
data = pd.read_csv("https://raw.githubusercontent.com/danang-ctrl/psds_3.0/main/data/Linear%20Regression%20-%20Sheet1.csv")
X = np.array(data['X'])
Y = np.array(data['Y'])
X_train, X_test,  y_train, y_test = train_test_split(X,Y)
a = lin_reg(X_train,y_train)
plt.scatter(X_train,y_train, label = 'data')
plt.plot(X_train,a.predict(),color = 'red', label = 'prediksi')
plt.legend()
plt.show()