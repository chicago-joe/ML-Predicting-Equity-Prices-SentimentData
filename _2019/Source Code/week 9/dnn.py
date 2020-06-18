from keras.models import Sequential
import tensorflow as tf
from keras.layers.core import Dense, Activation   
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import os
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from scipy.stats.mstats import winsorize
from keras import backend as K
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

train_x ="train_x.txt"
train_y ="train_y.txt"
test_x = "test_x.txt"
test_y ="test_y.txt"

train_x= pd.read_csv(train_x,delimiter=",",index_col=0)
train_y = pd.read_csv(train_y,delimiter=",",index_col=0,header=None)

test_x = pd.read_csv(test_x,delimiter=",",index_col=0)
test_y= pd.read_csv(test_y,delimiter=",",index_col=0,header=None)

scaler = StandardScaler()
scaler.fit(train_x)        # compute the mean and std dev which will be used below
X_scaled = scaler.transform(train_x)
X_test_scaled=scaler.transform(test_x)

table={}
for layer1 in range(10,100,10):
    for layer2 in range(10,100,10):
        for layer3 in range(10,100,10):
            model = Sequential()
            model.add(Dense(layer1, init='uniform', input_dim=15))
            model.add(Activation('linear'))
             
            model.add(Dense(layer2))
            model.add(Activation('linear'))
            
            model.add(Dense(layer3))
            model.add(Activation('linear'))
             
            model.add(Dense(1))
            model.add(Activation('linear'))

#20,50,20
#model.compile(loss='mean_squared_error', optimizer="adam", metrics=["accuracy",Accuracy_approx])
            sgd = SGD(lr=0.0001, decay=0.00001)
            model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy","mse"])
              
            out=model.predict(X_test_scaled)

            up_dir = 0
            down_dir = 0
            for i in range(len(out)):
                if (out[i]>0) and (test_y.iloc[i,0]>0):
                    up_dir += 1
                elif ((out[i]<0) and (test_y.iloc[i,0]<0)):
                    down_dir += 1
                else:
                    continue
            
            up_dir_y = 0
            down_dir_y = 0
            for i in test_y.iloc[:,0]:
                if i > 0:
                    up_dir_y += 1
                else:
                    down_dir_y += 1
            
            up_dir_pred = 0
            down_dir_pred = 0
            for i in range(len(out)):
                if out[i]>0:
                    up_dir_pred += 1
                else:
                    down_dir_pred += 1
            score=r2_score(test_y, out)
            table["layer1:%i"%layer1+" layer2:%i"%layer2+" layer3:%i"%layer3]=[up_dir,down_dir,up_dir_y,down_dir_y,up_dir_pred,down_dir_pred,score,out]


model = Sequential()
model.add(Dense(20, init='uniform', input_dim=15))
model.add(Activation('linear'))
 
model.add(Dense(20))
model.add(Activation('linear'))

model.add(Dense(60))
model.add(Activation('linear'))
 
model.add(Dense(1))
model.add(Activation('linear'))

sgd = SGD(lr=0.0001, decay=0.00001)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["accuracy","mse"])
 
hist=model.fit(X_scaled, train_y, batch_size=64, epochs=300, shuffle=True,verbose=0,validation_split=0.2)
plt.plot(hist.history['loss'])
plt.xlabel('epoch')
plt.ylabel('cost function')
plt.show()

model.evaluate(X_scaled, train_y, batch_size=10)
model.evaluate(X_test_scaled,test_y,batch_size=10)
out=model.predict(X_test_scaled)
out2=model.predict(X_scaled)
r2_score(test_y, out)                   
mean_squared_error(test_y, out) 

precision=[]
recall=[]
key=[]
accuracy=[]
r2=[]
epsilon=0.0000001
for i in table.keys():
    precision.append(table[i][0]/(table[i][4]+epsilon)+table[i][1]/(table[i][5]+epsilon))
    recall.append(table[i][0]/table[i][2]+table[i][1]/table[i][3])
    key.append(i)
    accuracy.append(table[i][0]+table[i][1])
    r2.append(table[i][6])
mp=max(precision)
mx=max(accuracy)
mr=max(recall)
mr2=max(r2)
for i in range(len(precision)):
    if (abs(mp-precision[i])<=0.01):
        best=key[i]
        print(table[best])

plt.scatter(out2,
            (out2 - train_y),
            c='steelblue',
            edgecolors = 'white',
            marker='o',
            s=35,
            alpha=0.9,
            label='Training data')
plt.scatter(out,
            (out - test_y),
            c='limegreen',
            edgecolors = 'white',
            marker='s',
            s=35,
            alpha=0.9,
            label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-0.075, xmax=0.075, lw=2, color='black')
plt.xlim([-0.075,0.075])
plt.show()