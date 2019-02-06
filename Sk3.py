#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 03:28:59 2019

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
 
df = pd.read_excel("SSandMH_clear_gps.xlsx")
df = df[df["Cluster"] != -1]
y = df["PerSq"]
X = df.drop(["ID", "Area", "PerSq", "Price", "Street", "Bedrooms", "Gas", "Storeroom", "Floors", "Parking", "Lat", "Lng", "District", "Rooms"], axis=1)
#X = pd.get_dummies(X, columns=["Status", "Condition", "District"], prefix=["St", "Co", "Di"], drop_first=True)
X = pd.get_dummies(X, columns=["Status", "Condition", "Cluster"], prefix=["STA", "CON", "CLA"], drop_first=True)

#spilt dataset
Xtrn, Xtest, Ytrn, Ytest = train_test_split(X, y, random_state=1, test_size=0.1)

 
#model = RandomForestRegressor(n_estimators=150, max_features='sqrt', n_jobs=-1)  # случайный лес
models = [LinearRegression(),
          RandomForestRegressor(n_estimators=100, max_features='sqrt'),
          KNeighborsRegressor(n_neighbors=6),
          SVR(kernel='linear', C=1e3),
          SVR(kernel='rbf', C=1e3, epsilon=0.2),
          LogisticRegression()
          ]

 
TestModels = pd.DataFrame()
tmp = {}
 
for model in models:
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