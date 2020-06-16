import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import pandas as pd

train_x ="train_x.txt"
train_y ="train_y.txt"
test_x = "test_x.txt"
test_y ="test_y.txt"
  
train_x= pd.read_csv(train_x,delimiter=",",index_col=0)
train_y= pd.read_csv(train_y,delimiter=",",index_col=0,header=None)

test_x= pd.read_csv(test_x,delimiter=",",index_col=0)
test_y= pd.read_csv(test_y,delimiter=",",index_col=0,header=None)

scaler = StandardScaler()
scaler.fit(train_x)   
X_scaled = scaler.transform(train_x)
X_test_scaled=scaler.transform(test_x)

table={}
for degree in range(1,3):
    print(degree)
    poly = PolynomialFeatures(degree)
    X_poly=poly.fit_transform(X_scaled)
    X_test_poly=poly.transform(X_test_scaled)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, train_y)
    out=pol_reg.predict(X_test_poly)
    
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
    table["degree: %i"%degree]=[score,up_dir,down_dir,up_dir_y,down_dir_y,up_dir_pred,down_dir_pred]

    print(r2_score(test_y, out))           
    print(MSE(test_y, out))
    print(up_dir+down_dir)

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
pred_y=out
test_y_df=test_y
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
pd.DataFrame(pred_y).to_csv('pred_y_pm.csv',sep=',')
