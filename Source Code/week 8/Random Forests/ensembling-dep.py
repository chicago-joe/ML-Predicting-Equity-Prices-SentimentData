################################################################
# ensembling-dep.py
# Ensemble Methods
# Created by Joseph Loss on 11/06/2019
#
# Contact: loss2@illinois.edu
###############################################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pylab as plot
import matplotlib.pyplot as plt
# from sklearn import ensemble
# from sklearn.pipeline import make_pipeline


address = "C:/Users/jloss/PyCharmProjects/SMA-HullTrading-Practicum/Source Code/Week 8/"
X_train = address + "train_x.txt"
y_train = address + "train_y.txt"
X_test = address + "test_x.txt"
y_test = address + "test_y.txt"

# read in-sample and out-sample datasets
y_train = pd.read_csv(y_train, sep = ',', header = None)
y_test = pd.read_csv(y_test, sep = ',', header = None)
X_train = pd.read_csv(X_train, sep = ',', header = 0)
X_test = pd.read_csv(X_test, sep = ',', header = 0)

# add column headers
column_names = ['Date', 'return']
y_train.columns = column_names
y_test.columns = column_names

# set Date index
y_train.set_index('Date', inplace = True)
y_test.set_index('Date', inplace = True)
X_train.set_index('Date', inplace = True)
X_test.set_index('Date', inplace = True)

# Preprocess / Standardize data
sc_X = StandardScaler()
X_train_std = sc_X.fit_transform(X_train)
X_test_std = sc_X.transform(X_test)

# now standardize the y
sc_y = StandardScaler()
y_train = np.array(y_train).reshape(-1,1)
y_train_std = sc_y.fit_transform(y_train)
y_test_std = sc_y.transform(y_test)

## Random Forests Model: Variance-Reduction Approach
names = X_train.columns.tolist()
featNames = np.array(names)

RFmodel = RandomForestRegressor(n_estimators = 1000,     # should be between 100 to 500
                                criterion = 'mse',
                                max_depth = 10,
                                max_features = "auto",
                                # oob_score = True,
                                n_jobs = -1,
                                random_state = None)      # Brieman and Cutler recommendation for regression problems

# fit model
y_train_std = np.ravel(y_train_std)
y_test_std = np.ravel(y_test_std)
RFmodel.fit(X_train_std, y_train_std)

# predict on in-sample and oos
y_train_std_pred = RFmodel.predict(X_train_std)
y_test_std_pred = RFmodel.predict(X_test_std)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train_std, y_train_std_pred),
        mean_squared_error(y_test_std, y_test_std_pred)))

print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train_std, y_train_std_pred),
        r2_score(y_test_std, y_test_std_pred)))

# RFmodel.score(X_test_std,y_train_std)
#
# # plot Feature Importance of RandomForests model
# featureImportance = RFmodel.feature_importances_
# featureImportance = featureImportance / featureImportance.max()    # scale by max importance
# sorted_idx = np.argsort(featureImportance)
# barPos = np.arange(sorted_idx.shape[0]) + 0.5
# plot.barh(barPos, featureImportance[sorted_idx], align = 'center')      # chart formatting
# plot.yticks(barPos, featNames[sorted_idx])
# plot.xlabel('Variable Importance')
# plot.show()

## SCATTER PLOTS: //TODO
plt.scatter(y_train_pred,
            (y_train_pred - y_train),
            c='steelblue',
            edgecolors = 'white',
            marker='o',
            s=35,
            alpha=0.9,
            label='Training data')
plt.scatter(y_test_pred,
            (y_test_pred - y_test),
            c='limegreen',
            edgecolors = 'white',
            marker='s',
            s=35,
            alpha=0.9,
            label='Test data')

plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.legend(loc='upper left')
plt.hlines(y=0, xmin=-4, xmax=5, lw=2, color='black')
plt.xlim([-4,5])
plt.show()



## Gradient Boosting Model: Error-Minimization Approach
# sc_X = StandardScaler()
# X_train_std = sc_X.fit_transform(X_train)
# X_test_std = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = np.array(y_train).reshape(-1,1)
# y_train_std = sc_y.fit_transform(y_train)
# y_test_std = sc_y.transform(y_test)

# GBmodel = GradientBoostingRegressor()

