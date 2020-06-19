# PCA.py
# Principal Component Analysis of the SPY Stationarity.txt data
#
# Created by Joseph Loss on 10/23/2019
# Contact: loss2@illinois.edu

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression

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

# filter for unstationary
features = df_daily_SPY.columns
for name in features:
    if df_stat_SPY[name].bool() == False:
        df_daily_SPY=df_daily_SPY.drop([name],axis=1)
df_stationary_final = df_daily_SPY

print('\n df_stationary head: ')
print(df_stationary_final.head())
# print(df_stationary_final.info())

X_train, X_test, y_train, y_test = train_test_split(df_stationary_final, target,
                                                    test_size=0.2,
                                                    random_state=42)


# before PCA we scale data such that each feature has unit variance:
scaler = StandardScaler()
scaler.fit(df_stationary_final)        # compute the mean and std dev which will be used below
X_scaled = scaler.transform(df_stationary_final)

# standardizing after train_test_split
sc = StandardScaler()
sc.fit_transform(X_train)
sc.transform(X_test)

n_pca_comps = 4

# check the min/max of the scaled featuress:
print("\n\nAfter scaling minimum: ", X_scaled.min(axis=0))
pca = PCA(n_components = n_pca_comps)
pca.fit(X_scaled)
X_pca=pca.transform(X_scaled)

# check the shape of the X_pca array
print("\nNumber of PCA Components: ", n_pca_comps)
print("X_pca shape: ", X_pca.shape)

# check the variance ratio of principal components,
# this shows us how much each component contributes to the total variance
ex_variance=np.var(X_pca, axis = 0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print("\n Principal Component Variance Ratio (% contributed to total variance): ")
print(ex_variance_ratio)

stationary_feature_names=df_stationary_final.columns

# These principal components are calculated only from features and no information from classes are considered.
# We can make a heat-plot to see how the features mixed up to create the components.
plt.figure(figsize=(20,15))

hm = sns.heatmap(pca.components_,
                 cbar=False,
                 annot=True,
                 square=True,
                 fmt='.2f',
                 annot_kws={'size': 13},
                 yticklabels=['1st Comp','2nd Comp','3rd Comp','4th Comp'],
                 xticklabels=stationary_feature_names)

plt.xticks(fontsize=12,rotation=50)

plt.matshow(pca.components_,cmap='viridis')
plt.yticks([0,1,2,3],['1st Comp','2nd Comp','3rd Comp','4th Comp'],fontsize=10)
plt.colorbar()
plt.xticks(range(len(stationary_feature_names)),stationary_feature_names,rotation=65,ha='left')
plt.figure(figsize=(20,20))
plt.show()


# use predictions from the principal components to
# use same matrix
# use in-sample
# fit to in-sample

