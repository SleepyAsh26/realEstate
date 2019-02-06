# Artificial Neural Network
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_excel("SSandMH.xlsx")
y_val = df["PerSq"]
x_data = df.drop(["ID", "Area", "PerSq", "Price", "Street"], axis=1)

#spilt dataset
x_data = pd.get_dummies(x_data, columns=["Status", "Condition", "District"], prefix=["St", "Co", "Di"])


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_val, test_size = 0.2, random_state = 0)

# Feature Scaling
"""
from sklearn.preprocessing import StandardScaler
scX = StandardScaler()
scy = StandardScaler()
X_train = scX.fit_transform(X_train)
X_test = scX.transform(X_test)
"""
from sklearn.preprocessing import MinMaxScaler
scaler_model = MinMaxScaler()
scaler_model.fit(X_train)
X_train=pd.DataFrame(scaler_model.transform(X_train),columns=X_train.columns,index=X_train.index)
scaler_model.fit(X_test)
X_test=pd.DataFrame(scaler_model.transform(X_test),columns=X_test.columns,index=X_test.index)



# Evaluating the ANN 
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from keras import regularizers
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras import metrics


def baseline_model():
    model = Sequential()
    model.add(Dense(10, activation="relu",input_dim=63))
    #model.add(Dropout(0.1))
    model.add(Dense(10, activation="relu", kernel_initializer='normal'))
    #model.add(Dropout(0.1))
    #model.add(Dense(10, activation="relu", kernel_initializer='normal'))
    #model.add(Dense(24, activation="relu", kernel_initializer='normal'))

    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=[metrics.mae])
    return model
model = baseline_model()
model.fit(X_train, y_train, epochs=500, batch_size=100)    
scores = model.evaluate(X_test, y_test)
print(scores)
y_pred  = model.predict(X_test)

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# evaluate model with standardized dataset
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(estimator, x_data, y_val, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))



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





"""
estimator = KerasRegressor(build_fn=baseline_model)
par = [n for n in range(10,101,10)]
parameters = {'batch_size': [10, 50, 100],
              'epochs': [100, 500, 1000],
              'units1': par,
              'units2': par,
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = estimator,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

"""