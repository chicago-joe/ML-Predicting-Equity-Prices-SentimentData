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
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score,roc_auc_score,accuracy_score,explained_variance_score
import pylab as plot
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV 

# from sklearn import ensemble
# from sklearn.pipeline import make_pipeline


train_x ="train_x.txt"
train_y ="train_y.txt"
#train_y ="train_y_class.txt"
test_x = "test_x.txt"
test_y ="test_y.txt"
#test_y ="test_y_class.txt"


# read in-sample and out-sample datasets
X_train = pd.read_csv(train_x,delimiter=",",index_col=0)
y_train = pd.read_csv(train_y,delimiter=",",index_col=0,header=None)

X_test = pd.read_csv(test_x,delimiter=",",index_col=0)
y_test = pd.read_csv(test_y,delimiter=",",index_col=0,header=None)
test_y_df=y_test

# Preprocess / Standardize data
sc_X = StandardScaler()
X_train_std = sc_X.fit_transform(X_train)
X_test_std = sc_X.transform(X_test)

#y_train = np.array(y_train).reshape(-1,1)

## Random Forests Model: Variance-Reduction Approach
names = X_train.columns.tolist()
featNames = np.array(names)

RFmodel = RandomForestRegressor(criterion = 'mse',
#RFmodel = RandomForestClassifier(criterion = 'gini',
                                max_features = "auto",
                                n_jobs = -1,
                                random_state = None)      # Brieman and Cutler recommendation for regression problems

# fit model
para = {'max_leaf_nodes':range(2,10,1),'n_estimators':range(1,201,10)}
    
#CV_forest = GridSearchCV(RFmodel,para,cv=6, n_jobs= 4,iid = True,refit= True,scoring='accuracy')
CV_forest = GridSearchCV(RFmodel,para,cv=6, n_jobs= 4,iid = True,refit= True,scoring='explained_variance')
CV_forest.fit(X_train_std, y_train)

print(CV_forest.best_params_)
best_leaf_nodes = CV_forest.best_params_['max_leaf_nodes']
best_n = CV_forest.best_params_['n_estimators']

###############################################################################################################################

train_x ="train_x.txt"
train_y ="train_y.txt"
#train_y ="train_y_class.txt"
test_x = "test_x.txt"
test_y ="test_y.txt"
#test_y ="test_y_class.txt"


# read in-sample and out-sample datasets
X_train = pd.read_csv(train_x,delimiter=",",index_col=0)
y_train = pd.read_csv(train_y,delimiter=",",index_col=0,header=None)

X_test = pd.read_csv(test_x,delimiter=",",index_col=0)
y_test = pd.read_csv(test_y,delimiter=",",index_col=0,header=None)

# Preprocess / Standardize data
sc_X = StandardScaler()
X_train_std = sc_X.fit_transform(X_train)
X_test_std = sc_X.transform(X_test)

y_train = np.array(y_train).reshape(-1,1)
y_test = np.array(y_test).reshape(-1,1)

#best_leaf_nodes = None
#best_n = 11

RFmodel = RandomForestRegressor(n_estimators = best_n,max_leaf_nodes=best_leaf_nodes,n_jobs=-1)
#RFmodel = RandomForestClassifier(n_estimators = best_n,max_leaf_nodes=best_leaf_nodes,n_jobs=-1)
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

#print('accuracy train: %.3f, test: %.3f' % (
#        accuracy_score(y_train, y_train_pred),
#        accuracy_score(y_test, y_test_pred)))

print('explanined variance train: %.3f, test: %.3f' % (
        explained_variance_score(y_train, y_train_pred),
        explained_variance_score(y_test, y_test_pred)))

# RFmodel.score(X_test_std,y_train_std)

# plot Feature Importance of RandomForests model
featureImportance = RFmodel.feature_importances_
featureImportance = featureImportance / featureImportance.max()    # scale by max importance
sorted_idx = np.argsort(featureImportance)
barPos = np.arange(sorted_idx.shape[0]) + 0.5
plot.barh(barPos, featureImportance[sorted_idx], align = 'center')      # chart formatting
plot.yticks(barPos, featNames[sorted_idx])
plot.xlabel('Variable Importance')
plot.show()


plt.scatter(y_train_pred.reshape(-1,1),
            (y_train_pred.reshape(-1,1) - y_train.reshape(-1,1)),
            c='steelblue',
            edgecolors = 'white',
            marker='o',
            s=35,
            alpha=0.9,
            label='Training data')
plt.scatter(y_test_pred.reshape(-1,1),
            (y_test_pred.reshape(-1,1) - y_test.reshape(-1,1)),
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
pred_y=y_test_pred
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
    


pd.DataFrame(pred_y).to_csv('pred_y_rf.csv',sep=',')
test_y_df.to_csv('test_y_df.csv',sep=',',header=None)
