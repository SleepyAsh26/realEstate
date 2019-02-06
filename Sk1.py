#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 04:10:16 2019

@author: sleepyash
"""

#imports
import numpy as np
import pandas as pd
import math

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


#import training dataset
df = pd.read_excel("SSandMH.xlsx")
y = df["PerSq"]
X = df.drop(["ID", "Area", "PerSq", "Price", "Street"], axis=1)
#spilt dataset
X = pd.get_dummies(X, columns=["Status", "Condition", "District"], prefix=["St", "Co", "Di"])
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

print('Training score: {}'.format(lr_model.score(X_train, y_train)))
print('Test score: {}'.format(lr_model.score(X_test, y_test)))

y_pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)

print('RMSE: {}'.format(rmse))



steps = [
    ('scalar', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),
    ('model', Ridge(alpha=10, fit_intercept=True))
]

ridge_pipe = Pipeline(steps)
ridge_pipe.fit(X_train, y_train)

print('Training Score: {}'.format(ridge_pipe.score(X_train, y_train)))
print('Test Score: {}'.format(ridge_pipe.score(X_test, y_test)))

y_pred = ridge_pipe.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)

print('RMSE: {}'.format(rmse))



def acc():
    preds = []
    for i in y_pred:
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





import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc('xtick', labelsize=30) 
matplotlib.rc('ytick', labelsize=30) 

fig, ax = plt.subplots(figsize=(50, 40))

plt.style.use('ggplot')
plt.plot(y_pred, y_test, 'ro')
plt.xlabel('Predictions', fontsize = 30)
plt.ylabel('Reality', fontsize = 30)
plt.title('Predictions x Reality on dataset Test', fontsize = 30)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()








