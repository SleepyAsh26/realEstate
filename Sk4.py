#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 14:09:23 2019

@author: sleepyash
"""
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
 
df = pd.read_excel("SSandMH_clear_gps.xlsx")
df = df[df["Cluster"] != -1]
y = df["PerSq"]
#X = df.drop(["ID", "Area", "PerSq", "Price", "Street", "Bedrooms"], axis=1)
X = df.drop(["ID", "Area", "PerSq", "Price", "Street", "Bedrooms", "Gas", "Storeroom", "Floors","Parking", "Lat", "Lng", "District"], axis=1)
#X = pd.get_dummies(X, columns=["Status", "Condition", "District"], prefix=["St", "Co", "Di"], drop_first=True)
X = pd.get_dummies(X, columns=["Status", "Condition", "Cluster"], prefix=["St", "Co", "CL"], drop_first=True)

#scaler = StandardScaler()
#X = scaler.fit_transform(X)
#
scaler_model = MinMaxScaler()
X = scaler_model.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.1)

#
# Fit regression model
print("RBF")
svr_rbf = SVR(kernel='rbf', C=1e3)
y_rbf = svr_rbf.fit(X_train, y_train).predict(X_test)
print('Training score: {}'.format(svr_rbf.score(X_train, y_train)))
print('Test score: {}'.format(svr_rbf.score(X_test, y_test)))

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
