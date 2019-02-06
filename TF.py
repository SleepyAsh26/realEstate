#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:15:38 2019

@author: sleepyash
"""
import pandas as pd
import tensorflow as tf


df = pd.read_excel("SSandMH_clear_gps.xlsx")
df = df[df["Cluster"] != -1]
y = df["PerSq"]
#X = df.drop(["ID", "Area", "PerSq", "Price", "Street", "Bedrooms"], axis=1)
X = df.drop(["ID", "Area", "PerSq", "Price", "Street", "Bedrooms", "Gas", "Storeroom", "Floors","Parking", "Lat", "Lng", "District"], axis=1)
#X = pd.get_dummies(X, columns=["Status", "Condition", "District"], prefix=["St", "Co", "Di"], drop_first=True)
X = pd.get_dummies(X, columns=["Status", "Condition", "Cluster"], prefix=["St", "Co", "CL"], drop_first=True)

#Scaling
#MinMax
from sklearn.preprocessing import MinMaxScaler
scaler_model = MinMaxScaler()
X_scaled = pd.DataFrame(scaler_model.fit_transform(X),columns=X.columns,index=X.index)

"""
#Standard
from sklearn.preprocessing import StandardScaler
scaler_model = StandardScaler()
X = scaler_model.fit_transform(X)
"""
#Split Train/Test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=1)

#Creating Feature Columns
feat_cols=[]
for cols in X:
    column=tf.feature_column.numeric_column(cols)
    feat_cols.append(column)
    
#The estimator model
#model = tf.estimator.DNNRegressor(feature_columns=feat_cols,hidden_units=[10,10])


#optimizer='RMSProp' ('Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD') activation_fn = tf.nn.relu
model = tf.estimator.DNNRegressor(feature_columns=feat_cols, hidden_units=[10,20], optimizer='RMSProp')#

#the input function
input_func = tf.estimator.inputs.pandas_input_fn(X_train,y_train,batch_size=100, num_epochs=500,shuffle=False)

#Training the model
model.train(input_fn=input_func,steps=500)

#Evaluating the model
#train_metrics=model.evaluate(input_fn=input_func,steps=1000)

#Now to predict values we do the following
pred_input_func=tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=10,num_epochs=1,shuffle=False)
preds = model.predict(input_fn=pred_input_func)

final_pred=[]
for pred in list(preds):
    final_pred.append(pred["predictions"])
    

def acc():
    preds = []
    for i in final_pred:
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
plt.plot(final_pred, y_test, 'ro')
plt.xlabel('Predictions', fontsize = 30)
plt.ylabel('Reality', fontsize = 30)
plt.title('Predictions x Reality on dataset Test', fontsize = 30)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.show()





