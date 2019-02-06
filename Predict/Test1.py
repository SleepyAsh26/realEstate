#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 10:15:38 2019

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

######Scaling
#MinMax

from sklearn.preprocessing import MinMaxScaler
scaler_model = MinMaxScaler()
scaler_model.fit(X_train)
X_train = pd.DataFrame(scaler_model.transform(X_train),columns=X_train.columns,index=X_train.index)
X_test = pd.DataFrame(scaler_model.transform(X_test),columns=X_test.columns,index=X_test.index)
"""
#Standard
from sklearn.preprocessing import StandardScaler
scaler_model = StandardScaler()
scaler_model.fit(X_train)
X_train = pd.DataFrame(scaler_model.transform(X_train),columns=X_train.columns,index=X_train.index)
X_test = pd.DataFrame(scaler_model.transform(X_test),columns=X_test.columns,index=X_test.index)
"""
#Creating Feature Columns
feat_cols=[]
for cols in x_data:
    column=tf.feature_column.numeric_column(cols)
    feat_cols.append(column)
    
#The estimator model
model = tf.estimator.DNNRegressor(feature_columns=feat_cols,hidden_units=[10,10])
#optimizer='RMSProp' ('Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD')
model = tf.estimator.DNNRegressor(feature_columns=feat_cols, 
                                  activation_fn = tf.nn.relu, 
                                  hidden_units=[10,10],
                                  optimizer='RMSProp')#

#tf.nn.elu
def leaky_relu(x):
    return tf.nn.relu(x) - 0.01 * tf.nn.relu(-x)
model = tf.estimator.DNNRegressor(feature_columns=feat_cols, 
                                  activation_fn = leaky_relu, 
                                  hidden_units=[200, 100, 50, 25, 12])#

#the input function
input_func=tf.estimator.inputs.pandas_input_fn(X_train,y_train,batch_size=100,num_epochs=500,shuffle=False)

#Training the model
model.train(input_fn=input_func,steps=500)


#Evaluating the model
#train_metrics=model.evaluate(input_fn=input_func,steps=1000)


#Now to predict values we do the following
pred_input_func=tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=1,num_epochs=100,shuffle=False)
preds=model.predict(input_fn=pred_input_func)

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













































































