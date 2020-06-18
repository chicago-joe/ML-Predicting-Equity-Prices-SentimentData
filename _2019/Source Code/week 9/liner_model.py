# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:28:47 2019

@author: Zenith
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

address = "./"
SPYstat = address + "train_x.txt"
name_target = address + "train_y.txt"

# convert txt to pandas DF
df_stat_SPY = pd.read_csv(SPYstat, delimiter = ",",index_col=0)
target = pd.read_csv(name_target, delimiter = ",",index_col=0,header=None)

# standardize the data
sc = StandardScaler()
sc.fit(df_stat_SPY)
df_daily_SPY = sc.transform(df_stat_SPY)

## ElasticNet Hyper Tuning
# Tuning grid set up
param_grid = {'alpha':np.linspace(0,0.005,10000),
              'l1_ratio':np.linspace(0,1,10)}

EN = ElasticNet()
clf = GridSearchCV(EN,
                   param_grid = param_grid,
                   cv = 10,
                   n_jobs = -1,
                   verbose = 5,
                   return_train_score = True)

clf.fit(df_daily_SPY,target)

result = pd.DataFrame.from_dict(clf.cv_results_)
result = result.sort_values(by = ['mean_test_score'],ascending = False)

result.to_csv(address + "grid result for EN.csv")

#########################################################################################################

train_x ="train_x.txt"
train_y ="train_y.txt"
test_x = "test_x.txt"
test_y ="test_y.txt"

train_x_df = pd.read_csv(train_x,delimiter=",",index_col=0)
train_y_df = pd.read_csv(train_y,delimiter=",",index_col=0,header=None)

test_x_df = pd.read_csv(test_x,delimiter=",",index_col=0)
test_y_df = pd.read_csv(test_y,delimiter=",",index_col=0,header=None)

scaler = StandardScaler()
scaler.fit(train_x_df)        # compute the mean and std dev which will be used below
X_scaled = scaler.transform(train_x_df)
X_test_scaled=scaler.transform(test_x_df)
a=result['param_alpha'].iloc[0]

l1=result['param_l1_ratio'].iloc[0]

#a=0.000707570757075707
#l1=1.0 
EN = ElasticNet(alpha = a,l1_ratio = l1)
EN.fit(train_x_df,train_y_df)
pred_y = EN.predict(test_x_df)
pred_train_y=EN.predict(train_x_df)

EN.score(test_x_df,test_y_df)
MSE(test_y_df,pred_y)


plt.scatter(pred_train_y,
            (pred_train_y - train_y_df[1]),
            c='steelblue',
            edgecolors = 'white',
            marker='o',
            s=35,
            alpha=0.9,
            label='Training data')
plt.scatter(pred_y,
            (pred_y - test_y_df[1]),
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


up_dir = 0
down_dir = 0
up_0_00=0
up_00_1=0
up_1_5=0
up_5=0
down_0_00=0
down_00_1=0
down_1_5=0
down_5=0
up_0_00_dir=0
up_00_1_dir=0
up_1_5_dir=0
up_5_dir=0
down_0_00_dir=0
down_00_1_dir=0
down_1_5_dir=0
down_5_dir=0

pre_up_0_00_dir=0
pre_up_00_1_dir=0
pre_up_1_5_dir=0
pre_up_5_dir=0
pre_down_0_00_dir=0
pre_down_00_1_dir=0
pre_down_1_5_dir=0
pre_down_5_dir=0
for i in range(len(pred_y)):
    
    if ((pred_y[i]>0) and (test_y_df.iloc[i,0]>0) and (test_y_df.iloc[i,0]<0.001)):
        up_0_00_dir += 1
    elif ((pred_y[i]>0) and (test_y_df.iloc[i,0]>0.001) and (test_y_df.iloc[i,0]<0.01)):
        up_00_1_dir += 1
    elif ((pred_y[i]>0) and (test_y_df.iloc[i,0]>0.01) and (test_y_df.iloc[i,0]<0.05)):
        up_1_5_dir += 1
    elif ((pred_y[i]>0) and (test_y_df.iloc[i,0]>0.05)):
        up_5_dir += 1
    elif ((pred_y[i]<0) and (test_y_df.iloc[i,0]<0) and (test_y_df.iloc[i,0]>-0.001)):
        down_0_00_dir += 1
    elif ((pred_y[i]<0) and (test_y_df.iloc[i,0]<-0.001) and (test_y_df.iloc[i,0]>-0.01)):
        down_00_1_dir += 1
    elif ((pred_y[i]<0) and (test_y_df.iloc[i,0]<-0.01) and (test_y_df.iloc[i,0]>-0.05)):
        down_1_5_dir += 1
    elif ((pred_y[i]<0) and (test_y_df.iloc[i,0]<-0.05)):
        down_5_dir += 1


    if ((pred_y[i]>0.001) and (test_y_df.iloc[i,0]>0) and (test_y_df.iloc[i,0]<0.001)):
        pre_up_0_00_dir += 1
    elif ((pred_y[i]>0.001) and (test_y_df.iloc[i,0]>0.001) and (test_y_df.iloc[i,0]<0.01)):
        pre_up_00_1_dir += 1
    elif ((pred_y[i]>0.001) and (test_y_df.iloc[i,0]>0.01) and (test_y_df.iloc[i,0]<0.05)):
        pre_up_1_5_dir += 1
    elif ((pred_y[i]>0.001) and (test_y_df.iloc[i,0]>0.05)):
        pre_up_5_dir += 1
    elif ((pred_y[i]<-0.001) and (test_y_df.iloc[i,0]<0) and (test_y_df.iloc[i,0]>-0.001)):
        pre_down_0_00_dir += 1
    elif ((pred_y[i]<-0.001) and (test_y_df.iloc[i,0]<-0.001) and (test_y_df.iloc[i,0]>-0.01)):
        pre_down_00_1_dir += 1
    elif ((pred_y[i]<-0.001) and (test_y_df.iloc[i,0]<-0.01) and (test_y_df.iloc[i,0]>-0.05)):
        pre_down_1_5_dir += 1
    elif ((pred_y[i]<-0.001) and (test_y_df.iloc[i,0]<-0.05)):
        pre_down_5_dir += 1

    
    if ((pred_y[i]>0) and (pred_y[i]<0.001) and (test_y_df.iloc[i,0]>0) and (test_y_df.iloc[i,0]<0.001)):
        up_0_00 += 1
    elif ((pred_y[i]>0.001) and (pred_y[i]<0.01) and (test_y_df.iloc[i,0]>0.001) and (test_y_df.iloc[i,0]<0.01)):
        up_00_1 += 1
    elif ((pred_y[i]>0.01) and (pred_y[i]<0.05) and (test_y_df.iloc[i,0]>0.01) and (test_y_df.iloc[i,0]<0.05)):
        up_1_5 += 1
    elif ((pred_y[i]>0.05) and (test_y_df.iloc[i,0]>0.05)):
        up_5 += 1
    elif ((pred_y[i]<0) and (pred_y[i]>-0.001) and (test_y_df.iloc[i,0]<0) and (test_y_df.iloc[i,0]>-0.001)):
        down_0_00 += 1
    elif ((pred_y[i]<-0.001) and (pred_y[i]>-0.01) and (test_y_df.iloc[i,0]<-0.001) and (test_y_df.iloc[i,0]>-0.01)):
        down_00_1 += 1
    elif ((pred_y[i]<-0.01) and (pred_y[i]>-0.05) and (test_y_df.iloc[i,0]<-0.01) and (test_y_df.iloc[i,0]>-0.05)):
        down_1_5 += 1
    elif ((pred_y[i]<-0.05)  and (test_y_df.iloc[i,0]<-0.05)):
        down_5 += 1

    
    if ((pred_y[i]>0) and (test_y_df.iloc[i,0]>0)):
        up_dir += 1
    elif ((pred_y[i]<0) and (test_y_df.iloc[i,0]<0)):
        down_dir += 1


up_dir_y = 0
down_dir_y = 0
up_0_00_y=0
up_00_1_y=0
up_1_5_y=0
up_5_y=0
down_0_00_y=0
down_00_1_y=0
down_1_5_y=0
down_5_y=0
for i in test_y_df.iloc[:,0]:
    if i>0 and i<0.001:
        up_0_00_y+=1
    elif i>0.001 and i<0.01:
        up_00_1_y+=1
    elif i>0.01 and i<0.05:
        up_1_5_y+=1
    elif i>0.05:
        up_5_y+=1
    elif i<0 and i>-0.001:
        down_0_00_y+=1 
    elif i<-0.001 and i>-0.01:
        down_00_1_y+=1 
    elif i<-0.01 and i>-0.05:
        down_1_5_y+=1 
    elif i<-0.05:
        down_5_y+=1 
    if i > 0:
        up_dir_y += 1
    else:
        down_dir_y += 1
        
pre_up_dir_y = 0
pre_down_dir_y = 0
pre_up_0_000_y=0
pre_up_000_00_y=0
pre_up_00_1_y=0
pre_up_1_5_y=0
pre_up_5_y=0
pre_down_0_000_y=0
pre_down_000_00_y=0
pre_down_00_1_y=0
pre_down_1_5_y=0
pre_down_5_y=0
for i in pred_y:
    if i>0 and i<0.0001:
        pre_up_0_000_y+=1
    elif i>0.0001 and i<0.001:
        pre_up_000_00_y+=1
    elif i>0.001 and i<0.01:
        pre_up_00_1_y+=1
    elif i>0.01 and i<0.05:
        pre_up_1_5_y+=1
    elif i>0.05:
        pre_up_5_y+=1
    elif i<0 and i>-0.0001:
        pre_down_0_000_y+=1 
    elif i<-0.0001 and i>-0.001:
        pre_down_000_00_y+=1 
    elif i<-0.001 and i>-0.01:
        pre_down_00_1_y+=1 
    elif i<-0.01 and i>-0.05:
        pre_down_1_5_y+=1 
    elif i<-0.05:
        pre_down_5_y+=1 
    if i > 0:
        pre_up_dir_y += 1
    else:
        pre_down_dir_y += 1
    
EN.score(test_x_df,test_y_df)
#EN.score(train_x_df,train_y_df)
pd.DataFrame(pred_y).to_csv('pred_y_liner.csv', sep=',')
