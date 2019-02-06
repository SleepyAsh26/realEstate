#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 17:51:24 2019

@author: sleepyash
"""

from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np

df=pd.read_excel("saburtalo.xlsx")
y = df["PerSq"]
X = df.drop(["ID", "Views","PerSq", "Price","District","Address"], axis=1)
#tf.set_random_seed(1)
# Encoding categorical data
X = pd.get_dummies(X, columns=["Status", "Condition", "Street"], prefix=["St", "Co","Str"])



Xtrn, Xtest, Ytrn, Ytest = train_test_split(X, y, random_state=0, test_size=0.1)
###
scaler_model = MinMaxScaler()
scaler_model.fit(Xtrn)
Xtrn=pd.DataFrame(scaler_model.transform(Xtrn),columns=Xtrn.columns,index=Xtrn.index)
scaler_model.fit(Xtest)
Xtest=pd.DataFrame(scaler_model.transform(Xtest),columns=Xtest.columns,index=Xtest.index)
###
"""
scX = StandardScaler()
Xtrn = scX.fit_transform(Xtrn)
Xtest = scX.transform(Xtest)
"""
#####
"""
model =  KNeighborsRegressor(n_jobs=-1)
estimators = np.arange(1, 100, 5)
scores = []
for n in estimators:
    model.set_params(n_neighbors=n)
    model.fit(Xtrn, Ytrn)
    scores.append(model.score(Xtest, Ytest))
plt.title("Effect of n_estimators")
plt.xlabel("n_estimator")
plt.ylabel("score")
plt.plot(estimators, scores)
plt.show()
"""
# model = RandomForestRegressor(n_estimators=150, max_features='sqrt', n_jobs=-1)  # случайный лес
models = [#LinearRegression(),
          #RandomForestRegressor(n_estimators=90, max_features='sqrt'),
          KNeighborsRegressor(n_neighbors=11),
          SVR(kernel='rbf', C=1e3, gamma=0.01),
          #SVR(kernel='poly', C=1e3, degree=2, epsilon=0.5),
          #SVR(kernel='linear', C=1e3),
          LogisticRegression()
          ]
 
TestModels = pd.DataFrame()
tmp = {}
 
for model in models:
    print("H")
    # get model name
    m = str(model)
    tmp['Model'] = m[:m.index('(')]
    # fit model on training dataset
    model.fit(Xtrn, Ytrn)
    # predict prices for test dataset and calculate r^2
    tmp['R2_Price'] = r2_score(Ytest, model.predict(Xtest))
    # write obtained data
    TestModels = TestModels.append([tmp])
 
TestModels.set_index('Model', inplace=True)
 
fig, axes = plt.subplots(ncols=1, figsize=(10, 4))
TestModels.R2_Price.plot(ax=axes, kind='bar', title='R2_Price')
plt.show()