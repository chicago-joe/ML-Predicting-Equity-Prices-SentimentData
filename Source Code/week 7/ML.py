# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 21:54:04 2019

@author: Zenith Zhou

"""

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score,accuracy_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression

# address = "D:/Github Repo/SMA-HullTrading-Practicum/Data/"
# SPYstat = address+"SPYstationarity.txt"
# SPYdaily = address+"SPYdaily.txt"
address = "C:/Users/jloss/PyCharmProjects/SMA-HullTrading-Practicum/Source Code/Week 7/"
SPYstat = address+"SPYstationarity.txt"
SPYdaily = address+"SPYdaily.txt"



# convert txt to pandas DF
df_stat_SPY = pd.read_csv(SPYstat,delimiter = ",")
df_daily_SPY = pd.read_csv(SPYdaily, delimiter = ",")

# clean empty cell
df_daily_SPY = df_daily_SPY.dropna()
df_daily_SPY.set_index('Date',inplace=True)

target = df_daily_SPY["next_Return"]

df_daily_SPY = df_daily_SPY.drop(["next_Return"],axis=1)
df_daily_SPY = df_daily_SPY.drop(["today_Return"],axis=1)
df_daily_SPY = df_daily_SPY.drop(["classret"],axis=1)

# filter for unstationary
features = df_daily_SPY.columns
for name in features:
    if df_stat_SPY[name].bool() == False:
        df_daily_SPY=df_daily_SPY.drop([name],axis=1)
        
X_train, X_test, y_train, y_test = train_test_split(df_daily_SPY, target, 
                                                    test_size=0.2,
                                                    random_state=42)

features = df_daily_SPY.columns

# Trending for Ridge
## 0-0.01 study
alpha = np.linspace(0,0.02,1000)
ridge_df = pd.DataFrame()
ridge_score = {}

for i in range(len(alpha)):
    ridge = Ridge(alpha=alpha[i])
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    ridge.transform(X_test, y_test)
    y_test_pred = ridge.predict(X_test)
    # y_test_pred = ridge.predict(X_test)

    ridge_df[float(alpha[i])] = ridge.coef_
    #print(ridge.score(X_train,y_train))
    ridge_score[float(alpha[i])] = ridge.score(X_train,y_train)

ridge_test_df = pd.DataFrame()
ridge_test_score = {}

for i in range(len(alpha)):
    # ridge = Ridge(alpha=alpha[i])
    ridge.transform(X_train, y_train)
    y_test_pred = ridge.predict(X_test)

    ridge_df[float(alpha[i])] = ridge.coef_
    #print(ridge.score(X_train,y_train))
    ridge_score[float(alpha[i])] = ridge.score(X_train,y_train)

ridge_df = ridge_df.set_index(features)
ridge_df = ridge_df.transpose()

# ridge_df.plot(logx=True,grid=True)
#
# plt.legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.)
# plt.title("Ridge With 0 - 0.02 Range")
# plt.ylabel("Coefficient")
# plt.xlabel("Lambda Value")
# plt.figure(figsize=(20,20))
#
# plt.plot(list(ridge_score.keys()),list(ridge_score.values()))
# plt.grid()
# plt.xlim(0)
# plt.title("Ridge score")
# plt.xlabel("Lambda")
# plt.ylabel("R^2")
# plt.figure(figsize=(20,20))

ridge_df.to_csv(address+"ridge_res.csv")

# Trending for Lasso
## 0-0.1 study
alpha = np.linspace(0,0.00002,1000)
lasso_df = pd.DataFrame()
lasso_score = {}

for i in range(len(alpha)):
    lasso = Lasso(alpha=alpha[i])
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)
    lasso_score[float(alpha[i])] = lasso.score(X_train,y_train)
    
    lasso_df[float(alpha[i])] = lasso.coef_

lasso_df = lasso_df.set_index(features)
lasso_df = lasso_df.transpose()

# lasso_df.plot(logx=True,grid=True)
#
# plt.legend(bbox_to_anchor=(1.05, 1), loc='best', borderaxespad=0.)
# plt.title("Lasso With 0 - 5e-5 Range")
# plt.ylabel("Coefficient")
# plt.xlabel("Lambda Value")
# plt.figure(figsize=(20,10))
#
# plt.plot(list(lasso_score.keys()),list(lasso_score.values()))
# plt.xticks(rotation=70)
# plt.grid()
# plt.xlim(0)
# plt.title("Ridge score")
# plt.xlabel("Lambda")
# plt.ylabel("R^2")
# plt.figure(figsize=(20,20))


lasso_df.to_csv(address+"lasso_res.csv")