#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 04:35:20 2019

@author: sleepyash
"""

import math
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# #############################################################################
df = pd.read_excel("SSandMH.xlsx")
y = df["PerSq"]
X = df.drop(["ID", "Area", "PerSq", "Price", "Street"], axis=1)
#spilt dataset
#Xtrn, Xtest, Ytrn, Ytest = train_test_split(training_dataset[analytics_fields], training_dataset[['price']],test_size=0.2)

X = pd.get_dummies(X, columns=["Status", "Condition", "District"], prefix=["St", "Co", "Di"])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)
# #############################################################################
scaler_model = MinMaxScaler()
scaler_model.fit(X_train)
X_train=pd.DataFrame(scaler_model.transform(X_train),columns=X_train.columns,index=X_train.index)
scaler_model.fit(X_test)
X_test=pd.DataFrame(scaler_model.transform(X_test),columns=X_test.columns,index=X_test.index)
######
scX = StandardScaler()
X_train = scX.fit_transform(X_train)
X_test = scX.transform(X_test)
###############################################################################
# Fit regression model
print("RBF")
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.01)
y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
print('Training score: {}'.format(svr_rbf.score(X_train, y_train)))
print('Test score: {}'.format(svr_rbf.score(X_test, y_test)))
print("Linear")
svr_lin = SVR(kernel='linear', C=1e3)
y_lin = svr_lin.fit(X_train, y_train).predict(X_test)
print('Training score: {}'.format(svr_lin.score(X_train, y_train)))
print('Test score: {}'.format(svr_lin.score(X_test, y_test)))
print("Poly")
svr_poly = SVR(kernel='poly', C=1e3, degree=2, epsilon=0.5)
y_poly = svr_poly.fit(X_train, y_train).predict(X_test)
print('Training score: {}'.format(svr_poly.score(X_train, y_train)))
print('Test score: {}'.format(svr_poly.score(X_test, y_test)))
#mse = mean_squared_error(y_test, y_rbf)
#rmse = math.sqrt(mse)


from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

acc_score = []
kf = KFold(n_splits=3)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    svr_rbf.fit(X_train,y_train)
    predictions = svr_rbf.predict(X_test)
    
    acc_score.append(accuracy_score(predictions, y_rbf))

np.mean(acc_score)




def acc():
    preds = []
    for i in y_rbf:
        preds.append(i)
    test = list(y_test.values)
    compare=[]
    for i in range(0,len(test)):
        compare.append((test[i]/preds[i])*100)
    good = 0
    for i in compare:
        if i <120 and i >90:
            good += 1
    print(good,len(compare))
    res = round(good/len(compare)*100,2)
    print(res,end="%")
    return res
acc = acc()


matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 
fig, ax = plt.subplots(figsize=(50, 40))
plt.style.use('ggplot')
plt.plot(y_rbf, y_test, 'ro')
plt.xlabel('Predictions', fontsize = 30)
plt.ylabel('Reality', fontsize = 30)
plt.title('Predictions x Reality on dataset Test', fontsize = 30)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()





















df=pd.read_excel("Data2.xlsx")
Test_y = df["PerSq"]
Test_X = df.drop(["ID", "Area", "Views","PerSq", "Price", "Street"], axis=1)
Test_X = pd.get_dummies(Test_X, columns=["Status", "Condition", "District"], prefix=["St", "Co", "Di"])

Test_pred = svr_rbf.predict(Test_X)

def acc():
    preds = []
    for i in Test_pred:
        preds.append(i)
    test = list(Test_y.values)
    compare=[]
    for i in range(0,len(test)):
        compare.append((test[i]/preds[i])*100)
    good = 0
    for i in compare:
        if i <110 and i >90:
            good += 1
    print(good,len(compare))
    res = round(good/len(compare)*100,2)
    print(res,end="%")
    return res
acc = acc()


matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 

fig, ax = plt.subplots(figsize=(50, 40))

plt.style.use('ggplot')
plt.plot(Test_pred, Test_y, 'ro')
plt.xlabel('Predictions', fontsize = 30)
plt.ylabel('Reality', fontsize = 30)
plt.title('Predictions x Reality on dataset Test', fontsize = 30)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()


