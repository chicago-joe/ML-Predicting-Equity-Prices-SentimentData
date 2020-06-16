# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 21:10:57 2019

@author: Zenith Zhou
"""


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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


address = "C:/Users/yz_ze/Documents/GitHub/SMA-HullTrading-Practicum/Data/"
SPYstat = address + "SPYstationarity.txt"
SPYdaily = address + "SPYdaily.txt"

# convert txt to pandas DF
df_stat_SPY = pd.read_csv(SPYstat, delimiter = ",")
df_daily_SPY = pd.read_csv(SPYdaily, delimiter = ",")

# clean empty cell
df_daily_SPY = df_daily_SPY.dropna()
df_daily_SPY.set_index('Date', inplace = True)

target = df_daily_SPY["next_Return"]
df_daily_SPY = df_daily_SPY.drop(["next_Return"], axis = 1)
df_daily_SPY = df_daily_SPY.drop(["today_Return"], axis = 1)

# filter for unstationary
features = df_daily_SPY.columns
for name in features:
    if df_stat_SPY[name].bool() == False:
        df_daily_SPY = df_daily_SPY.drop([name], axis = 1)

sc = StandardScaler()
sc.fit(df_daily_SPY)
df_daily_SPY = sc.transform(df_daily_SPY)

param_grid = {'alpha':np.linspace(0,0.005,10000),
              'l1_eatio':np.linspace(0,1,10)}

EN = ElasticNet()
clf = GridSearchCV(EN,
                   param_grid = param_grid,
                   cv = 10,
                   n_jobs = -1,
                   verbose = 5,
                   return_train_score = True)

clf.fit(df_daily_SPY)

result = pd.DataFrame.from_dict(clf.cv_results_)
result = result.sort_values(by = ['mean_test_score'],ascending = False)

result.to_csv("grid result for EN.csv")