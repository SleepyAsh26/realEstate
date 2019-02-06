#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 04:13:03 2018

@author: sleepyash
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df=pd.read_csv("d.csv")
y_val = df["PerSq"]
#x_data = df.drop(['ID', 'Area', 'Views', 'PerSq'], axis=1)
x_data = df.drop(['Area', 'PerSq'], axis=1)
#Encoding
x_data = pd.get_dummies(x_data, columns=["Status", "Condition", "District"], prefix=["St", "Co", "Di"])

#Split
X_train, X_test, y_train, y_test = train_test_split(x_data,y_val,test_size=0.1,random_state=101)

#Scale
scaler_model = MinMaxScaler()
scaler_model.fit(X_train)
X_train=pd.DataFrame(scaler_model.transform(X_train),columns=X_train.columns,index=X_train.index)
scaler_model.fit(X_test)
X_test=pd.DataFrame(scaler_model.transform(X_test),columns=X_test.columns,index=X_test.index)

#Creating Feature Columns
feat_cols=[]
for cols in x_data:
    column=tf.feature_column.numeric_column(cols)
    feat_cols.append(column)
"""
#LOOP FOR TESTING EVERY VALUE
layers=[]
file = open("layers.txt","w")
#for i in range(0,101,10):   
#    if i > 0:
#        layers.append(i)
for j in range(20,101,10):
    layers.append(j)
    for k in range(10,101,10):
        layers.append(k)
        print(layers)
                

        res =round(good/len(compare)*100,2)
        print(res)
        file.write("Layers: "+"".join(str(layers))+"\n")
        file.write("Result: "+str(res)+"\n")
        layers=layers[:len(layers)-1]
    #layers=layers[:len(layers)-1]
    layers.clear()
file.close() 
print("END!!!!!!!!!")
"""
#The estimator model                         39,34  40,30
model=tf.estimator.DNNRegressor(hidden_units=[40,30],feature_columns=feat_cols)

#the input function
input_func=tf.estimator.inputs.pandas_input_fn(X_train,y_train,batch_size=500,num_epochs=1000,shuffle=False)

#Training the model
model.train(input_fn=input_func,steps=1000)

#Evaluating the model
#train_metrics=model.evaluate(input_fn=input_func,steps=100)

#Now to predict values we do the following
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test,y=y_test,batch_size=10,num_epochs=1,shuffle=False)
preds = model.predict(input_fn=pred_input_func)

real = y_test
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


#ADD DATA TO PREDICT
Pdf = pd.read_csv("newData2.csv")
price = Pdf["PerSq"]
info = Pdf.drop("PerSq",axis=1)
info = Pdf.drop("ID",axis=1)
info = Pdf.drop("Views",axis=1)
#Encoding
info = pd.get_dummies(info, columns=["Status", "Condition", "District"], prefix=["St", "Co", "Di"])
scaler_model.fit(info)
info = pd.DataFrame(scaler_model.transform(info),columns=info.columns,index=info.index)
#predict
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=info,y=price,batch_size=10,num_epochs=1,shuffle=False)
results = model.predict(input_fn=pred_input_func)
results = list(results)
Predictions=[]
for pred in results:
    Predictions.append(pred["predictions"])


file = open("layers.txt","r")
file = file.read().replace("\n",'').split("Layers: ")
res = open("res.txt","w")
for i in file:
    k = i[-4:]
    res.write(k+"\n")
res.close()
"""

"""
res = open("res.txt","w")
for i in final_pred:
    i = i.tolist()
    res.write(str(round(i[0]))+"\n")
res.close()
"""
#x_data.to_csv('export_dataframe.csv', float_format='%.3f', sep='\t')
"""
# Save the weights
model.save_weights('/')

# Restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')


loss,acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))

saver = tf.train.Saver()
sess = tf.Session()
sess.run(tf.local_variables_initializer())
saver.save(sess, 'my-model', global_step=1000) 
"""



