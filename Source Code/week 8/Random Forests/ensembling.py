################################################################
# ensembling.py
# Ensemble Methods
# Created by Joseph Loss on 11/06/2019
#
# Contact: loss2@illinois.edu
###############################################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pylab as plot
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV 



import os
os.chdir('/Users/jloss/PyCharmProjects/SMA-HullTrading-Practicum/Source Code/week 8/')
train_x ="train_x.txt"
train_y ="train_y.txt"
test_x = "test_x.txt"
test_y ="test_y.txt"


# read in-sample and out-sample datasets
y_train = pd.read_csv("train_y.txt", sep = ',', header = None)
y_test = pd.read_csv("test_y.txt", sep = ',', header = None)
X_train = pd.read_csv("train_x.txt", sep = ',', header = 0)
X_test = pd.read_csv("test_x.txt", sep = ',', header = 0)

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

# y_train = np.array(y_train).reshape(-1,1)
y_train = np.ravel(y_train)


## Random Forests Model: Variance-Reduction Approach
names = X_train.columns.tolist()
featNames = np.array(names)

RFmodel = RandomForestRegressor(n_estimators = 100,     # should be between 100 to 500
                                criterion = 'mse',
                                max_depth = 1,
                                max_features = "auto",
                                # n_jobs = -1,
                                random_state = None)      # Brieman and Cutler recommendation for regression problems

# fit model
para = {'max_depth':range(1,15), 'min_samples_leaf':range(1,20)}

CV_forest = GridSearchCV(RFmodel,para,cv=6, n_jobs = 1, iid = True,refit= True)
CV_forest.fit(X_train_std, y_train)

print(CV_forest.best_params_)
best_leaf = CV_forest.best_params_['min_samples_leaf']
best_depth = CV_forest.best_params_['max_depth']

RFmodel = RandomForestRegressor(random_state=None,min_samples_leaf=best_leaf,max_depth=best_depth,n_jobs=-1)
RFmodel.fit(X_train_std, y_train)

# predict on in-sample and oos
y_train_pred = RFmodel.predict(X_train_std)
y_test_pred = RFmodel.predict(X_test_std)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))

print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_pred),
        r2_score(y_test, y_test_pred)))


# plot Feature Importance of RandomForests model
featureImportance = RFmodel.feature_importances_
featureImportance = featureImportance / featureImportance.max()    # scale by max importance
sorted_idx = np.argsort(featureImportance)
barPos = np.arange(sorted_idx.shape[0]) + 0.5

plot.barh(barPos, featureImportance[sorted_idx], align = 'center')      # chart formatting
plot.yticks(barPos, featNames[sorted_idx])
plot.xlabel('Variable Importance')
plot.show()

# param config
up_dir = 0
down_dir = 0
for i in range(len(y_test_pred)):
    if ((y_test_pred[i]>0) and (y_test.iloc[i,0]>0)):
        up_dir += 1
    elif ((y_test_pred[i]<0) and (y_test.iloc[i,0]<0)):
        down_dir += 1
    else:
        continue

up_dir_y = 0
down_dir_y = 0
for i in y_test.iloc[:,0]:
    if i > 0:
        up_dir_y += 1
    else:
        down_dir_y += 1

up_dir_pred = 0
down_dir_pred = 0
for i in range(len(y_test_pred)):
    if y_test_pred[i]>0:
        up_dir_pred += 1
    else:
        down_dir_pred += 1


# scatter plots
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

