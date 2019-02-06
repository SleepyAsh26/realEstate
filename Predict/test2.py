#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 16:05:07 2019

@author: sleepyash
"""

import pandas as pd
import tensorflow as tf
df = pd.read_excel("SSandMH.xlsx")
y_val = df["PerSq"]
x_data = df.drop(["ID", "Area", "PerSq", "Price", "Street"], axis=1)
#spilt dataset
#Xtrn, Xtest, Ytrn, Ytest = train_test_split(training_dataset[analytics_fields], training_dataset[['price']],test_size=0.2)

x_data = pd.get_dummies(x_data, columns=["Status", "Condition", "District"], prefix=["St", "Co", "Di"])


#Split Train/Test
from sklearn.model_selection import train_test_split
X_train, X_test,y_train,y_test=train_test_split(x_data,y_val,test_size=0.1,random_state=1)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
X_train = scX.fit_transform(X_train)
X_test = scX.transform(X_test)
"""
from sklearn.preprocessing import MinMaxScaler
scaler_model = MinMaxScaler()
scaler_model.fit(X_train)
X_train = pd.DataFrame(scaler_model.transform(X_train),columns=X_train.columns,index=X_train.index)
X_test = pd.DataFrame(scaler_model.transform(X_test),columns=X_test.columns,index=X_test.index)

"""

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
score = regressor.score(X_test,y_test)



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

























