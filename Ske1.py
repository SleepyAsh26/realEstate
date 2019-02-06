#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 02:26:40 2019

@author: sleepyash
"""
from sklearn.svm import SVR
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
 
df = pd.read_excel("SSandMH.xlsx")
df = df.drop(["ID", "PerSq", "Street"], axis=1)
#df = pd.get_dummies(df, columns=["Status", "Condition", "District"], prefix=["St", "Co", "Di"])


def get_outliners(dataset, outliers_fraction=0.25):
    clf = svm.OneClassSVM(nu=0.95 * outliers_fraction + 0.05, kernel="rbf", gamma=0.1)
    clf.fit(dataset)
    result = clf.predict(dataset)
    print("")
    return result

training_dataset = df[get_outliners(df, 0.15)==1]

X = training_dataset.loc[:, df.columns != 'Price']
scaler_model = MinMaxScaler()
X = scaler_model.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, training_dataset[['Price']],random_state=1, test_size=0.2)
                                            
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
"""
 
# model = RandomForestRegressor(n_estimators=150, max_features='sqrt', n_jobs=-1)  # случайный лес
models = [LinearRegression(),
          RandomForestRegressor(n_estimators=100, max_features='sqrt'),
          KNeighborsRegressor(n_neighbors=6),
          SVR(kernel='linear'),
          LogisticRegression()
          ]
 
TestModels = pd.DataFrame()
tmp = {}
 
for model in models:
    # get model name
    m = str(model)
    tmp['Model'] = m[:m.index('(')]
    # fit model on training dataset
    model.fit(X_train, y_train)
    # predict prices for test dataset and calculate r^2
    tmp['R2_Price'] = r2_score(y_test, model.predict(X_test))
    # write obtained data
    TestModels = TestModels.append([tmp])
 
TestModels.set_index('Model', inplace=True)
 
fig, axes = plt.subplots(ncols=1, figsize=(10, 4))
TestModels.R2_Price.plot(ax=axes, kind='bar', title='R2_Price')
plt.show()
"""