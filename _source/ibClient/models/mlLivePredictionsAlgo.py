# --------------------------------------------------------------------------------------------------
# ML_model_LIVE.py
#
#
# --------------------------------------------------------------------------------------------------
# created by Joseph Loss on 11/01/2020

ticker = 'SPY'
max_n = 2500
# testStart = '2017-12-31'
testStart = '2017-12-10'
nTestDays = 100             # prediction day = n + 1
seed = 42
wait_time = 0

LOG_LEVEL = 'INFO'

# --------------------------------------------------------------------------------------------------
# Module imports

import numpy as np
np.random.seed(seed = seed)

import os, logging, time
from datetime import timedelta
from datetime import datetime as dt
from talib import SMA, ATR
import warnings
warnings.simplefilter("ignore")

from fastai.tabular.all import *
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from scipy.stats.mstats import winsorize
from statsmodels.tsa.stattools import adfuller

import matplotlib.pyplot as plt
import pylab as plot
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot as qp

# custom imports
from fnCommon import setPandas, fnUploadSQL, setOutputFilePath, setLogging, fnOdbcConnect

LOG_FILE_NAME = os.path.basename(__file__)


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

def predict(df, nTrain, nTest, dfS):

    X_train = df[0:nTrain]
    X_test = df[nTrain:nTrain + nTest]

    y_train = X_train['rtnTodayToTomorrow']
    y_test = X_test['rtnTodayToTomorrow']

    # drop y variables from features
    X_train.drop(['rtnTodayToTomorrow', 'rtnTodayToTomorrowClassified'], axis = 1,)
    X_test.drop(['rtnTodayToTomorrow', 'rtnTodayToTomorrowClassified'], axis = 1)


    # --------------------------------------------------------------------------------------------------
    # winsorize / feature scaling

    X_train = X_train.apply(winsorizeData, axis = 0)
    maxTrain = X_train.max()
    minTrain = X_train.min()

    conditions = [(X_test.values < minTrain.values), (X_test.values > maxTrain.values)]
    choices = [minTrain, maxTrain]
    tmp = np.select(conditions, choices, default = X_test)

    X_test = pd.DataFrame._from_arrays(tmp.transpose(), columns = X_test.columns, index = X_test.index)


    # --------------------------------------------------------------------------------------------------
    # test for stationarity

    stationarityResults = (stationarity(X_train))
    stationarityResults = pd.DataFrame(stationarityResults, index = [0])
    stationaryFactors = []

    for i in stationarityResults.columns:
        if stationarityResults[i][0] == 1:
            stationaryFactors.append(i)

    X_test = X_test[stationaryFactors].drop(['rtnTodayToTomorrow',
                          'rtnTodayToTomorrowClassified', ]
                         ,
                         axis = 1)
    X_train = X_train[stationaryFactors].drop(['rtnTodayToTomorrow',
                            'rtnTodayToTomorrowClassified', ]
                           ,
                           axis = 1)


# --------------------------------------------------------------------------------------------------
    # Preprocess / Standardize data

    sc_X = StandardScaler()
    X_train_std = sc_X.fit_transform(X_train)
    X_test_std = sc_X.fit_transform(X_test)

    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)


    # --------------------------------------------------------------------------------------------------
    # init random forest

    RFmodel = RandomForestRegressor(criterion = 'mse',
                                    max_features = "auto",
                                    n_estimators = max_n,
                                    # max_leaf_nodes = 2,
                                    # oob_score = True,
                                    # max_depth=10,
                                    min_samples_leaf = 100,
                                    random_state = seed,
                                    n_jobs = -1)

    RFmodel.fit(X_train_std, y_train)

    temp = pd.Series(RFmodel.feature_importances_, index = X_test.columns)
    dfFeat[df.index[-1]] = temp

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

    ## quantile is super common number so subtract .000001 or have quantile >= to the signal
    signal_Q1 = np.quantile(predictionsY, 0.25) - 0.000001
    dfS.at[df.index[-1], 'signal_Q1'] = signal_Q1


    last_signal = predictionsY[-1]
    dfS.at[df.index[-1], 'last_signal'] = last_signal
    dfS.at[df.index[-1], 'signal_rolling_Q1'] = (dfS['last_signal'] - 0.000001).rolling(window = 7).quantile(0.25).fillna(9999)

    if dfS.at[df.index[-1], 'signal_rolling_Q1'] == 9999:
        dfS.at[df.index[-1], 'position'] = 0

    else:
        if (last_signal > 0.0001) & (last_signal > signal_Q1) & (last_signal < 0.001):
            dfS.at[df.index[-1], 'position'] = 1.0

        elif 0.001 > last_signal > 0.0001:
            dfS.at[df.index[-1], 'position'] = 0.75

        elif (last_signal > 0.001) & (dfS.at[df.index[-1], 'signal_rolling_Q1'] < last_signal) & (dfS.at[df.index[-1], 'signal_Q1'] < dfS.at[df.index[-1], 'signal_rolling_Q1']):
            dfS.at[df.index[-1], 'position'] = -1

        # elif (last_signal < 0.0009 ) & (dfS.at[df.index[-1], 'signal_rolling_Q1'] > dfS.at[df.index[-1], 'signal_Q1']*2):
        #     dfS.at[df.index[-1], 'position'] = -1

        elif last_signal > dfS.at[df.index[-1],'signal_rolling_Q1'] + dfS.at[df.index[-1], 'signal_Q1']:
            dfS.at[df.index[-1], 'position'] = 1

        else:
            dfS.at[df.index[-1], 'position'] = 0

    logging.info('Last signal:\t %.6f' % dfS.at[df.index[-1], 'last_signal'].astype(float))
    logging.info('Q1 signal:\t\t%.6f' % dfS.at[df.index[-1], 'signal_Q1'].astype(float))
    logging.info('Rolling signal:\t %.6f' % dfS.at[df.index[-1], 'signal_rolling_Q1'].astype(float))
    logging.info('Current Position:\t %.2f' % dfS.at[df.index[-1], 'position'].squeeze().astype(float))

    time.sleep(wait_time)
    featureImportances = pd.DataFrame(dfFeat)

    return [dfS, featureImportances]


# --------------------------------------------------------------------------------------------------
# compute portfolio returns with position sizing

def fnComputePortfolioRtn(dfStk= None, pos = None):
    dfP = pos.copy()

    # set to time-aware index
    dfP.index = pd.DatetimeIndex(dfP.index.get_level_values(0))
    dfP.index = pd.DatetimeIndex(dfP.index)

    dfU = dfStk.copy()
    dfU['return_T'] = dfU['adjClose'].pct_change().shift(-1)
    dfU.index = pd.DatetimeIndex(dfU.index)
    dfP = dfP.merge(dfU[['adjClose', 'return_T']], how = 'left', left_index = True, right_index = True)

    # calculate cumulative asset return
    dfP['creturn_T'] = ((1 + dfP['return_T']).cumprod() - 1).fillna(0)

    # calculate cumulative portfolio return
    dfP['return_P'] = dfP['position'] * dfP['return_T']
    dfP['creturn_P'] = (1 + dfP['return_P']).cumprod() - 1

    print('\n\n----------------------------------------')
    print('----------------------------------------')
    logging.info('Starting Portfolio:\n%s' % dfP.head(5))
    print('\n\n----------------------------------------')
    logging.info('Ending Portfolio:\n%s' % dfP.tail(5))
    print('\n----------------------------------------')

    return dfP


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# run main

if __name__ == '__main__':

    # custom pandas settings
    setPandas()
    setLogging(LOG_FILE_NAME = LOG_FILE_NAME, level = LOG_LEVEL)

    path = '_source\\'

    # set numpy float format
    floatFormatter = "{:,.6f}".format
    np.set_printoptions(formatter = {'float_kind':floatFormatter})


    # --------------------------------------------------------------------------------------------------

    try:

        q = """
                SELECT * 
                FROM smadb.tbllivepredictionfeatures
                
            """

        conn = fnOdbcConnect('smadb')
        dfAgg = pd.read_sql_query(q, conn)

        conn.disconnect()
        conn.close()

        # --------------------------------------------------------------------------------------------------
        # calculate predictions based on rolling model (refit rolling 100 days)

        # dfAgg = dfAgg.loc[dfAgg.index >= '2015-10-21']
        testDays = len(dfAgg.loc[dfAgg.index >= testStart])
        rollSet = dfAgg.loc[dfAgg.index < testStart]

        # set nTest (rolling every x days)
        nTrain = len(rollSet[:-nTestDays])
        nTest = nTestDays + 1

        dfS = pd.DataFrame(index = [dfAgg.loc[dfAgg.index >= testStart].index],
                           columns = ['signal_Q1', 'signal_rolling_Q1', 'last_signal', 'position'])

        dpred = { }
        dfFeat = { }


        for i in range(472, len(dfAgg) - nTrain - nTest, 1):
            print(dfAgg[i:nTrain+nTest+i])
            dpred, dfFeat = predict(dfAgg[i:nTrain + nTest + i], nTrain, nTest, dfS)

        dpred = dpred.iloc[:-1]


        # --------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------
        # close logger / handlers

        logging.info("========== END PROGRAM ==========")

    except Exception as e:
        logging.error(str(e), exc_info=True)

    # CLOSE LOGGING
    for handler in logging.root.handlers:
        handler.close()
    logging.shutdown()
