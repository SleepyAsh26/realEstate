#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 07:36:44 2018

@author: sleepyash
"""
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_excel('Data.xlsx')
y = dataset["Price"]
X = dataset.drop(['Price',"Address","Views", "ID"], axis=1)

# Encoding categorical data
X = pd.get_dummies(X, columns=["Status", "Condition", "District"], prefix=["St", "Co", "Di"])


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
X_train = scX.fit_transform(X_train)
X_test = scX.transform(X_test)


#df = df = pd.DataFrame(X)
#df.to_csv('export_dataframe.csv', float_format='%.3f', sep='\t')


from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
kf.get_n_splits(X)

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

"""
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

"""
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
regressor = LinearRegression()
regressor.fit(X_train, y_train)

SVRregressor = SVR(kernel='poly', C=1e3, degree=2)#gamma=0.05)
SVRregressor.fit(X_train, y_train)
y_pred = SVRregressor.predict(X_test)
score = SVRregressor.score(X_test,y_test)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
score = regressor.score(X_test,y_test)
#


# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(regressor, open(filename, 'wb'))


# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
accuracy = loaded_model.score(X, y)
print(accuracy)
result = loaded_model.predict(X)

#Procentage
y_pred = abs(y_pred)

compare = ((y_pred/y_test))*100
good = []
bad = []
for i in compare:
    if i <120 and i >90:
        good.append(i)
    else:
        bad.append(i)
print(len(bad))
print(len(good))
print(round(len(good)/len(compare)*100,2),end='%')

#np.savetxt("foo.csv", compare, delimiter=",")











