#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 22:49:08 2018

@author: sleepyash
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df=pd.read_excel("Data.xlsx")
y_val = df["Price"]
x_data = df.drop(['Price',"Address","Views", "ID"], axis=1)

# Encoding categorical data
x_data = pd.get_dummies(x_data, columns=["Status", "Condition", "District"], prefix=["St", "Co", "Di"])


X_train, X_eval,y_train,y_eval=train_test_split(x_data,y_val,test_size=0.2,random_state=0)

#Scaling
scaler_model = MinMaxScaler()
scaler_model.fit(X_train)
X_train=pd.DataFrame(scaler_model.transform(X_train),columns=X_train.columns,index=X_train.index)
scaler_model.fit(X_eval)
X_eval=pd.DataFrame(scaler_model.transform(X_eval),columns=X_eval.columns,index=X_eval.index)


#Creating Feature Columns
feat_cols=[]
for cols in x_data:
    column=tf.feature_column.numeric_column(cols)
    feat_cols.append(column)
#('Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD')
model=tf.estimator.LinearRegressor(feature_columns=feat_cols,optimizer='RMSProp')
#The estimator model
model=tf.estimator.DNNRegressor(feature_columns=feat_cols,hidden_units=[40, 30], optimizer='RMSProp')

#the input function
input_func=tf.estimator.inputs.pandas_input_fn(X_train,y_train,batch_size=100,num_epochs=1000,shuffle=False)


#Training the model
model.train(input_fn=input_func,steps=1000)


#Evaluating the model
#train_metrics=model.evaluate(input_fn=input_func,steps=1000)


#Now to predict values we do the following
pred_input_func=tf.estimator.inputs.pandas_input_fn(x=X_eval,y=y_eval,batch_size=1,num_epochs=1,shuffle=False)
preds=model.predict(input_fn=pred_input_func)

    
real = y_eval
#def proc(pred,real):
predictions=list(preds)
final_pred=[]
for pred in predictions:
    final_pred.append(pred["predictions"])

res = []
for pred in final_pred:
    res.append(int(pred))
compare = ((res/real))*100
good = 0
for i in compare:
    if i <120 and i >90:
        good += 1

print(good,len(compare))
print(round(good/len(compare)*100,2),end="%")



