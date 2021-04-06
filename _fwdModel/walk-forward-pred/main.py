# --------------------------------------------------------------------------------------------------
# backtest_v3,py
#
#
# --------------------------------------------------------------------------------------------------
# created by Joseph Loss on 4/5/2021

ticker = 'SPY'
LOG_LEVEL = 'INFO'

# --------------------------------------------------------------------------------------------------
# Module imports

import os, logging, time
import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime as dt
import warnings
warnings.simplefilter("ignore")

from fastai.tabular.all import *
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from scipy.stats.mstats import winsorize
from statsmodels.tsa.stattools import adfuller

# custom imports
# sys.path.append('..\\..\\_source')
from fnLibrary import setPandas, fnUploadSQL, setOutputFilePath, setLogging, fnOdbcConnect
LOG_FILE_NAME = os.path.basename(__file__)


# --------------------------------------------------------------------------------------------------
# walk forward model

def fnWalkForward(df=None, targetVar=None, nTest=None, nTrain=100, winsorize=True, stationary=True, preprocessing=True, **modelParams):

    # set numpy float format
    floatFormatter = "{:,.6f}".format
    np.set_printoptions(formatter = {'float_kind':floatFormatter})

    # set nTest (rolling every x days)
    nTest = nTest + 1

    dpred = { }
    dfLast = { }

    for i in range(0, len(df) - nTrain - nTest, 1):
        dpred, dfLast = predict(
                df[i:nTrain + nTest + i],
                nTrain,
                nTest,
                targetVar,
                winsorize,
                stationary,
                preprocessing,
                **modelParams)

    dpred = dpred.iloc[:-1]

    return dpred


# --------------------------------------------------------------------------------------------------
# winsorize data method

def winsorizeData(s):
    return winsorize(s, limits = [0.005, 0.005])


# --------------------------------------------------------------------------------------------------
# adf testing

def adf_test(timeSeries):
    dfADF = adfuller(timeSeries, autolag = 'AIC')
    output = pd.Series(dfADF[0:4], index = ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for key, value in dfADF[4].items():
        output['Critical Value (%s)' % key] = value
    logging.debug('ADF Testing: %s \n%s\n' % (timeSeries.name, output))

    return output


# --------------------------------------------------------------------------------------------------
# test stationarity

def stationarity(result):
    plist = { }
    for col in result:
        if adf_test(result[col])['p-value'] < 0.05:
            st = True
        else:
            st = False
        plist[col] = st

    return plist


# --------------------------------------------------------------------------------------------------
# predict

def predict(df, nTrain, nTest, targetVar, winsorize, stationary, preprocessing, **modelParams):

    if modelParams['random_state'] is None:
        pass
    else:
        np.random.seed(seed = modelParams['random_state'])

    X_train = df[0:nTrain]
    X_test = df[nTrain:nTrain + nTest]

    y_train = X_train[targetVar]
    y_test = X_test[targetVar]

    # drop y variables from features
    X_train.drop(targetVar, axis = 1,)
    X_test.drop(targetVar, axis = 1)


    # --------------------------------------------------------------------------------------------------
    # winsorize / feature scaling

    if winsorize:
        X_train = X_train.apply(winsorizeData, axis = 0)
        maxTrain = X_train.max()
        minTrain = X_train.min()

        conditions = [(X_test.values < minTrain.values), (X_test.values > maxTrain.values)]
        choices = [minTrain, maxTrain]
        tmp = np.select(conditions, choices, default = X_test)

        X_test = pd.DataFrame._from_arrays(tmp.transpose(), columns = X_test.columns, index = X_test.index)

    else:
        X_train = X_train
        X_test = X_test


    # --------------------------------------------------------------------------------------------------
    # test for stationarity

    if stationary:
        stationarityResults = (stationarity(X_train))
        stationarityResults = pd.DataFrame(stationarityResults, index = [0])
        stationaryFactors = []

        for i in stationarityResults.columns:
            if stationarityResults[i][0] == 1:
                stationaryFactors.append(i)

        X_test = X_test[stationaryFactors].drop(targetVar,
                                                axis = 1)
        X_train = X_train[stationaryFactors].drop(targetVar,
                                                  axis = 1)
    else:
        X_test = X_test.drop(targetVar,
                             axis = 1)
        X_train = X_train.drop(targetVar,
                               axis = 1)


# --------------------------------------------------------------------------------------------------
    # Preprocess / Standardize data

    X_train.loc[(X_train.values==-np.inf)|(X_train.values==np.inf)|(X_train.values==np.nan)]

    if preprocessing:

        sc_X = StandardScaler()
        X_train_std = sc_X.fit_transform(X_train)
        X_test_std = sc_X.fit_transform(X_test)

        y_train = np.array(y_train).reshape(-1, 1)
        y_test = np.array(y_test).reshape(-1, 1)


    # --------------------------------------------------------------------------------------------------
    # init random forest

    RFmodel = RandomForestRegressor(criterion = modelParams['rf_criterion'],
                                    max_features = modelParams['max_features'],
                                    n_estimators = modelParams['n_estimators'],
                                    # max_leaf_nodes = modelParams['max_leaf_nodes'],
                                    # oob_score = modelParams['oob_score'],
                                    # max_depth = modelParams['max_depth'],
                                    min_samples_leaf = modelParams['min_samples_leaf'],
                                    # min_samples_leaf = 100,
                                    random_state = modelParams['random_state'],
                                    n_jobs = modelParams['n_jobs'])

    RFmodel.fit(X_train_std, y_train)

    temp = pd.Series(RFmodel.feature_importances_, index = X_test.columns)
    # dfFeat[df.index[-1]] = temp

    # predict in-sample and out-of-sample
    y_train_pred = RFmodel.predict(X_train_std)
    y_test_pred = RFmodel.predict(X_test_std)


    # --------------------------------------------------------------------------------------------------
    # statistics

    print('\n\n----------------------------------------\n')
    logging.info("DATE: %s " % (pd.to_datetime(df.index[len(df) - 1]).date()))

    logging.info('MSE - Train: %.6f, Test: %.6f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))

    logging.info('R^2 - Train: %.4f, Test: %.4f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))

    logging.info('Explained Variance - Train: %.4f, Test: %.4f' % (
            explained_variance_score(y_train, y_train_pred),
            explained_variance_score(y_test, y_test_pred)))

    predictionsY = y_test_pred
    last_signal = predictionsY[-1]

    return [predictionsY, last_signal]


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# run main

if __name__ == '__main__':

    # custom pandas settings
    setPandas()
    setLogging(LOG_FILE_NAME = LOG_FILE_NAME, level = LOG_LEVEL)

    path = '..\\'

    # set numpy float format
    floatFormatter = "{:,.6f}".format
    np.set_printoptions(formatter = {'float_kind':floatFormatter})


    # --------------------------------------------------------------------------------------------------
    # TESTING

    df = pd.read_pickle('data.pkl')
    df.drop(columns = ['rtnTodayToTomorrowClassified'], inplace = True)

    # --------------------------------------------------------------------------------------------------

    try:

        model_parameters = {
                'rf_criterion':    'mse',
                'max_features':    'auto',
                'n_estimators':    1000,
                'min_samples_leaf':100,
                'random_state':    42,
                'n_jobs':          -1,
                # 'max_leaf_nodes':2,
                # 'oob_score':True,
                # 'max_depth':10
        }

        fnWalkForward(df,
                      targetVar = 'rtnTodayToTomorrow',
                      nTest = 100,
                      nTrain = 489,
                      winsorize = False,
                      stationary = True,
                      preprocessing = True,
                      **model_parameters)

        logging.info("========== END PROGRAM ==========")

    except Exception as e:
        logging.error(str(e), exc_info=True)

    # CLOSE LOGGING
    for handler in logging.root.handlers:
        handler.close()
    logging.shutdown()
