################################################################
# backtest.py
#
# Ensemble Methods
#
# Created by Joseph Loss on 6/22/2020
# --------------------------------------------------------------------------------------------------
# Module imports

import os, sys, logging
import copy
from pathlib import Path
from datetime import datetime as dt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score, explained_variance_score
from sklearn.model_selection import GridSearchCV
from scipy.stats.mstats import winsorize
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pylab as plot

import warnings
warnings.simplefilter("ignore")


# --------------------------------------------------------------------------------------------------
# custom imports

from fnCommon import setPandas, setLogging, setOutputFile
LOG_FILE_NAME = os.path.basename(__file__)


# --------------------------------------------------------------------------------------------------
# todo:
# classify returns

def fnClassifyReturns(rtnBins=None, stdDevBins = None, rtn=None, rtnStdDev=None):

    if rtnBins is None:
        # rtnBins = [2, 1, -1, 2]
        rtnBins = []
    else:
        rtnBins = rtnBins

    if stdDevBins is None:
        # stdDevBins = [1, 0.05, -0.05, -1]
        stdDevBins = []
    else:
        stdDevBins = stdDevBins

    if rtn > (rtnStdDev * stdDevBins[0]):
        rtnTodayToTomorrowClassified = 2
    elif rtn > (rtnStdDev * stdDevBins[1]):
        rtnTodayToTomorrowClassified = 1
    elif rtn > (rtnStdDev * stdDevBins[2]):
        rtnTodayToTomorrowClassified = 0
    elif rtn > (rtnStdDev * stdDevBins[3]):
        rtnTodayToTomorrowClassified = -1
    else:
        rtnTodayToTomorrowClassified = -2

    return rtnTodayToTomorrowClassified


# --------------------------------------------------------------------------------------------------
# pull in SPY prices to calculate returns today / tomorrow and bin them

# noinspection DuplicatedCode
def fnComputeReturns(ticker='SPY'):

    path = 'C:\\Users\\jloss\\PyCharmProjects\\ML-Predicting-Equity-Prices-SentimentData\\_source\\_data\\spyPrices\\'

    # todo:
    # dfSPY = pd.read_csv(path + "{}.csv".format(ticker))

    df = pd.read_csv(path + "SPY Price Data.csv")

    df.index = df['Date']
    df.index.name='date'
    df.sort_index(inplace=True)

    df['rtnTodayToTomorrow'] = df['Adj_Close'].pct_change().shift(-1)
    df['rtnYesterdayToToday'] = df['Adj_Close'].pct_change()

    # rtnTodayToTomorrow = (df['Adj_Close'][:-1].values - df['Adj_Close'][1:]) / df['Adj_Close'][1:]
    # rtnToday = (df['Adj_Close'][:-1] - df['Adj_Close'][1:].values) / df['Adj_Close'][1:].values

    rtnTodayToTomorrow = df['Adj_Close'].pct_change().shift(-1)
    rtnYesterdayToToday = df['Adj_Close'].pct_change()

    # compute rolling 250 day standard deviation
    rtnStdDev = rtnYesterdayToToday.iloc[::1].rolling(250).std().iloc[::1]
    rtnStdDev = rtnStdDev.dropna()
    rtnStdDev = rtnStdDev[1:]

    # classify returns tomorrow based on std deviation * bin
    rtnTodayToTomorrowClassified = [2 if rtnTodayToTomorrow[date] > rtnStdDev[date] * 1.0
                     else 1 if rtnTodayToTomorrow[date] > rtnStdDev[date] * 0.05
                     else 0 if rtnTodayToTomorrow[date] > rtnStdDev[date] * -0.05
                     else -1 if rtnTodayToTomorrow[date] > rtnStdDev[date] * -1.0
                     else -2 for date in rtnStdDev.index]

    # rtnTodayToTomorrowClassified = fnClassifyReturns(rtnBins=[2,1,0,-1,-2],stdDevBins = [1.0,0.05,-0.05,-1],rtn=rtnTodayToTomorrow, rtnStdDev = rtnStdDev)
    # rtnTodayToTomorrowClassified=[ 2  if ret>s2 else 1 if ret>s1 else 0 if ret>s0 else -1 if ret>sn1 else -2 for ret in rtnTodayToTomorrow]

    rtnTodayToTomorrowClassified = pd.DataFrame(rtnTodayToTomorrowClassified)
    rtnTodayToTomorrowClassified.index = rtnStdDev.index
    rtnTodayToTomorrowClassified.columns = ['rtnTodayToTomorrowClassified']


    # classify returns today based on std deviation * bin
    rtnYesterdayToTodayClassified = [2 if rtnYesterdayToToday[date] > rtnStdDev[date] * 1.0
                          else 1 if rtnYesterdayToToday[date] > rtnStdDev[date] * 0.05
                          else 0 if rtnYesterdayToToday[date] > rtnStdDev[date] * -0.05
                          else -1 if rtnYesterdayToToday[date] > rtnStdDev[date] * -1.0
                          else -2 for date in rtnStdDev.index]

    # rtnTodayToTomorrowClassified=[ 2  if ret>s2 else 1 if ret>s1 else 0 if ret>s0 else -1 if ret>sn1 else -2 for ret in rtnTodayToTomorrow]

    rtnYesterdayToTodayClassified = pd.DataFrame(rtnYesterdayToTodayClassified)
    rtnYesterdayToTodayClassified.index = rtnStdDev.index
    rtnYesterdayToTodayClassified.columns = ['rtnYesterdayToTodayClassified']

    rtnTodayToTomorrow = pd.DataFrame(rtnTodayToTomorrow)
    rtnYesterdayToToday = pd.DataFrame(rtnYesterdayToToday)
    rtnTodayToTomorrow.columns = ['rtnTodayToTomorrow']
    rtnYesterdayToToday.columns = ['rtnYesterdayToToday']

    return rtnYesterdayToToday, rtnTodayToTomorrow, rtnTodayToTomorrowClassified, rtnYesterdayToTodayClassified,rtnStdDev


# --------------------------------------------------------------------------------------------------
# read in activity feed data

def fnLoadActivityFeed(ticker='SPY'):

    path = 'C:\\Users\\jloss\\PyCharmProjects\\ML-Predicting-Equity-Prices-SentimentData\\_source\\_data\\activityFeed\\'

    colNames = ['ticker', 'date', 'description', 'sector', 
                'industry', 'raw_s', 's-volume', 's-dispersion', 
                'raw-s-delta', 'volume-delta', 'center-date', 
                'center-time', 'center-time-zone']

    dfSpy2015 = pd.read_csv(path + '{}2015.txt'.format(ticker), skiprows = 6, sep = '\t', names = colNames)
    dfSpy2016 = pd.read_csv(path + '{}2016.txt'.format(ticker), skiprows = 6, sep = '\t', names = colNames)
    dfSpy2017 = pd.read_csv(path + '{}2017.txt'.format(ticker), skiprows = 6, sep = '\t', names = colNames)
    dfSpy2018 = pd.read_csv(path + '{}2018.txt'.format(ticker), skiprows = 6, sep = '\t', names = colNames)
    dfSpy2019 = pd.read_csv(path + '{}2019.txt'.format(ticker), skiprows = 6, sep = '\t', names = colNames)

    # aggregating data
    df_temp = dfSpy2015.append(dfSpy2016, ignore_index = True)
    df_temp = df_temp.append(dfSpy2017, ignore_index = True)
    df_temp = df_temp.append(dfSpy2018, ignore_index = True)
    df_temp = df_temp.append(dfSpy2019, ignore_index = True)
    
    df_datetime = df_temp['date'].str.split(' ', n = 1, expand = True)
    df_datetime.columns = ['Date', 'Time']
    
    # merge datetime and aggregate dataframe
    dfAgg = pd.merge(df_temp, df_datetime, left_index = True, right_index = True)
    
    # filtering based on trading hours and excluding weekends
    dfAgg['Date'] = pd.to_datetime(dfAgg['Date'])
    dfAgg = dfAgg.loc[(dfAgg['Date'].dt.dayofweek != 5) & (dfAgg['Date'].dt.dayofweek != 6)]
    dfAgg = dfAgg[(dfAgg['Time'] >= '09:30:00') & (dfAgg['Time'] <= '16:00:00')]
    
    # excluding weekends
    # removing empty columns
    dfAgg = dfAgg.dropna(axis = 'columns')
    dfAgg = dfAgg.drop(columns = ['ticker', 'date', 'description', 'center-date', 
                            'center-time', 'center-time-zone', 'raw-s-delta', 'volume-delta'])
    
    # compute volume-base-s and ewm-volume-base-s
    dfAgg["volume_base_s"] = dfAgg["raw_s"] / dfAgg["s-volume"]
    dfAgg["ewm_volume_base_s"] = dfAgg.groupby("Date")["volume_base_s"].apply(lambda x:x.ewm(span = 390).mean())

    # aggregate by date
    dfT = dfAgg.groupby('Date').last().reset_index()
    dfT.index = dfT['Date']

    # compute factors
    dfT["mean_volume_base_s"] = dfAgg.groupby("Date")["volume_base_s"].mean()
    dfT["mean_raw_s"] = dfAgg.groupby("Date")["raw_s"].mean()
    dfT["mean_s_dispersion"] = dfAgg.groupby("Date")["s-dispersion"].mean()
    
    if ticker=='SPY':
        dfT['volume_base_s_z'] = (dfT['mean_volume_base_s'] - dfT['mean_volume_base_s'].rolling(26).mean()) \
                             / dfT['mean_volume_base_s'].rolling(26).std()
        dfT['s_dispersion_z'] = (dfT['mean_s_dispersion'] - dfT['mean_s_dispersion'].rolling(26).mean()) \
                            / dfT['mean_s_dispersion'].rolling(26).std()
    
    elif ticker == 'ES_F':
        dfT['volume_base_s_delta']=(dfT['mean_volume_base_s'][1:]-dfT['mean_volume_base_s'][:-1].values)
        dfT['s_dispersion_delta']=(dfT['mean_s_dispersion'][1:]-dfT['mean_s_dispersion'][:-1].values)
    
    dfT['raw_s_MACD_ewma12-ewma26'] = dfT["mean_raw_s"].ewm(span = 12).mean() - dfT["mean_raw_s"].ewm(span = 26).mean()

    dfT = dfT.drop(columns = ['Date', 'raw_s', 's-volume', 's-dispersion', 'Time', 'volume_base_s'])
    dfT.columns = ticker + ':' + dfT.columns

    return dfT


# --------------------------------------------------------------------------------------------------
# combine and aggregate spy / futures activity feed ata

def fnAggActivityFeed(df1, df2):

    dfA = pd.concat([df1, df2], axis = 1, sort = False)

    # pull Spy returns, classified tommorrow returns, classified today returns
    rtnYesterdayToToday, rtnTodayToTomorrow, rtnTodayToTomorrowClassified, rtnYesterdayToTodayClassified, rtnStdDev = fnComputeReturns(ticker='SPY')

    # rtnStdDev = rtnYesterdayToToday.iloc[::1].rolling(250).std().iloc[::1]
    # rtnStdDev = rtnStdDev.dropna()
    # rtnStdDev = rtnStdDev[1:]

    rtnStdDev.name = 'rtnStdDev'

    rtnStdDev.index.name='Date'
    rtnTodayToTomorrow.index.name='Date'
    rtnYesterdayToToday.index.name='Date'
    rtnTodayToTomorrowClassified.index.name='Date'
    rtnYesterdayToTodayClassified.index.name='Date'


    dfA.dropna(inplace=True)
    rtnTodayToTomorrow.dropna(inplace=True)
    rtnYesterdayToToday.dropna(inplace=True)

    dfAgg = pd.merge(dfA,rtnTodayToTomorrow,how='inner',left_index=True,right_index=True)
    dfAgg = pd.merge(dfAgg,rtnYesterdayToToday,how='inner',left_index=True,right_index=True)
    dfAgg = pd.merge(dfAgg,rtnTodayToTomorrowClassified,how='inner',left_index=True,right_index=True)
    dfAgg = pd.merge(dfAgg,rtnStdDev,how='inner',left_index=True,right_index=True)
    dfAgg = pd.merge(dfAgg,rtnYesterdayToTodayClassified,how='inner',left_index=True,right_index=True)

    return dfAgg


# --------------------------------------------------------------------------------------------------
# winsorize data method

def winsorizeData(s):
    return winsorize(s, limits = [0.005, 0.005])


# --------------------------------------------------------------------------------------------------
# adf testing

def adf_test(timeSeries):
    # print('Results of Augment Dickey-Fuller Test:')

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
# predictions

def predict(df, nTrain, nTest):

    X_train = df[0:nTrain]
    X_test = df[nTrain:nTrain + nTest]

    y_train = X_train['rtnTodayToTomorrow']
    y_test = X_test['rtnTodayToTomorrow']

    #    train_y_class=df_train['rtnTodayToTomorrowClassified']
    #    test_y_class=df_test['rtnTodayToTomorrowClassified']

    X_train.drop(['rtnTodayToTomorrow', 'rtnTodayToTomorrowClassified'], axis = 1)
    X_test.drop(['rtnTodayToTomorrow', 'rtnTodayToTomorrowClassified'], axis = 1)

    X_train = X_train.apply(winsorizeData, axis = 0)
    maxTrain = X_train.max()
    minTrain = X_train.min()

    conditions = [(X_test < minTrain), (X_test > maxTrain)]
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

    stationaryFactors.remove("rtnTodayToTomorrowClassified")
    stationaryFactors.remove("rtnTodayToTomorrow")

    logging.debug('Final Stationary Factors:\n%s' % pd.Series(stationaryFactors))

    X_train = X_train[stationaryFactors]
    X_test = X_test[stationaryFactors]


    # --------------------------------------------------------------------------------------------------
    # Preprocess / Standardize data

    sc_X = StandardScaler()
    X_train_std = sc_X.fit_transform(X_train)
    X_test_std = sc_X.transform(X_test)

    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    #
    # ## Random Forests Model: Variance-Reduction Approach
    # names = X_train.columns.tolist()
    # featNames = np.array(names)
    #
    # RFmodel = RandomForestRegressor(criterion = 'mse',
    #                                 # RFmodel = RandomForestClassifier(criterion = 'gini',
    #                                 max_features = "auto",
    #                                 n_jobs = -1,
    #                                 random_state = None)  # Brieman and Cutler recommendation for regression problems
    #
    # # fit model
    # para = { 'max_leaf_nodes':range(2, 3, 1), 'n_estimators':range(1, 501, 50) }
    #
    # CV_forest = GridSearchCV(RFmodel, para, cv = 6, n_jobs = -1, iid = True, refit = True, scoring = 'r2')
    # CV_forest.fit(X_train_std, y_train.ravel())
    #
    # print(CV_forest.best_params_)
    # best_leaf_nodes = CV_forest.best_params_['max_leaf_nodes']
    # best_n = CV_forest.best_params_['n_estimators']

    best_leaf_nodes = 2
    best_n = 401

    RFmodel = RandomForestRegressor(n_estimators = best_n, max_leaf_nodes = best_leaf_nodes, n_jobs = -1)
    # RFmodel = RandomForestClassifier(n_estimators = best_n,max_leaf_nodes=best_leaf_nodes,n_jobs=-1)

    RFmodel.fit(X_train_std, y_train)


    # --------------------------------------------------------------------------------------------------
    # predict in-sample and out-of-sample

    y_train_pred = RFmodel.predict(X_train_std)
    y_test_pred = RFmodel.predict(X_test_std)

    # print([df.index[len(df) - 1], y_test_pred])
    logging.info('Calculating %s predictions...' % df.index[len(df) - 1])


    # --------------------------------------------------------------------------------------------------
    # statistics

    print('\n ----------------------------------------')
    print('DATE: %s' % pd.to_datetime(df.index[len(df)-1]).date())
    print('MSE \n Train: %.3f, Test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))

    print('R^2 \n Train: %.3f, Test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))

    print('Explained Variance \n Train: %.3f, Test: %.3f' % (
            explained_variance_score(y_train, y_train_pred),
            explained_variance_score(y_test, y_test_pred)))


    # plot Feature Importance of RandomForests model
    # featureImportance = RFmodel.feature_importances_
    # featureImportance = featureImportance / featureImportance.max()  # scale by max importance
    # sorted_idx = np.argsort(featureImportance)
    # barPos = np.arange(sorted_idx.shape[0]) + 0.5
    # plot.barh(barPos, featureImportance[sorted_idx], align = 'center')  # chart formatting
    # plot.yticks(barPos, featNames[sorted_idx])
    # plot.xlabel('Variable Importance')
    # plot.show()
    #
    # plt.scatter(y_train_pred.reshape(-1, 1),
    #             (y_train_pred.reshape(-1, 1) - y_train.reshape(-1, 1)),
    #             c = 'steelblue',
    #             edgecolors = 'white',
    #             marker = 'o',
    #             s = 35,
    #             alpha = 0.9,
    #             label = 'Training data')
    # plt.scatter(y_test_pred.reshape(-1, 1),
    #             (y_test_pred.reshape(-1, 1) - y_test.reshape(-1, 1)),
    #             c = 'limegreen',
    #             edgecolors = 'white',
    #             marker = 's',
    #             s = 35,
    #             alpha = 0.9,
    #             label = 'Test data')
    #
    # plt.xlabel('Predicted values')
    # plt.ylabel('Residuals')
    # plt.legend(loc = 'upper left')
    # plt.hlines(y = 0, xmin = -0.075, xmax = 0.075, lw = 2, color = 'black')
    # plt.xlim([-0.075, 0.075])
    # plt.show()
    #
    return [df.index[len(df) - 1], y_test_pred]


# --------------------------------------------------------------------------------------------------
# determine firstQuantileRisk to spy based of 1st quantile signal

def firstQuantileRisk(value, q1signal):
    if value > q1signal:
        return 1
    elif value > 0:
        return 0.75
    else:
        return 2


# --------------------------------------------------------------------------------------------------
# determine the position based off predictionsY

def fnCreatePositions(predictionsY, positionRecord):

    index_y = predictionsY[0]
    predictionsY = predictionsY[1]

    q1signal = np.quantile(predictionsY, 0.25) - 0.000001
    lastsignal = predictionsY[-1]

    print('\n ')
    print('q1 signal: %s' % q1signal.round(6))
    print('last signal: %s' % lastsignal.round(6))
    print('\n ----------------------------------------')

    riskToSpy = (len(predictionsY) - 1) / sum([firstQuantileRisk(n, q1signal) for n in predictionsY])

    if ((lastsignal > q1signal) & (lastsignal > 0.000001)):
        return [index_y, riskToSpy]

    elif (lastsignal > 0.000001):
        return [index_y, riskToSpy * 0.75]

    elif (lastsignal > 0) & (lastsignal < 0.000001):
        return [index_y, 0.0]

    else:
        return [index_y, -1.0]


# --------------------------------------------------------------------------------------------------
# compute portfolio returns with position sizing

def fnComputePortfolioRtn(pred, pos):

    pos.columns=['position']

    # merge signal with position
    dfP = pd.merge(pos, pred, how='inner', left_index=True, right_index=True)

    # set to time-aware index
    dfP.index = pd.DatetimeIndex(dfP.index)

    # import SPY returns
    path = 'C:\\Users\\jloss\\PyCharmProjects\\ML-Predicting-Equity-Prices-SentimentData\\_source\\_data\\spyPrices\\'

    dfU = pd.read_csv(path + "SPY Price Data.csv")
    dfU.index = dfU['Date']
    dfU.sort_index(inplace=True)

    # merge with adjusted close price
    dfP = dfP.merge(dfU.Adj_Close, how = 'left', left_index = True, right_index = True)

    # merge percent return (including Day 0)
    dfU['rtnSPY'] = dfU['Adj_Close'].shift(-1).pct_change()[1:]
    dfP = dfP.merge(dfU.rtnSPY, how = 'left', left_index = True, right_index = True)

    # calculate cumulative asset return
    dfP['cRtn-SPY'] = ((1 + dfP['rtnSPY']).cumprod() - 1).fillna(0)

    # calculate position using rolling quantile bin
    dfP = fnPositionBinning(dfP, posBins=[1.0, 0.75, -1.0])

    # calculate cumulative portfolio return
    dfP['rtnPort'] = dfP['position'] * dfP['rtnSPY']
    dfP['cRtn-Port'] = (1 + dfP['rtnPort']).cumprod()-1

    cols = ['signal',
            'Adj_Close',
            'rtnSPY',
            'cRtn-SPY',
            'position',
            'rtnPort',
            'cRtn-Port']
    dfP = dfP[cols]

    return dfP


# --------------------------------------------------------------------------------------------------
#  position binning

def fnPositionBinning(df, posBins=[1.0, 0.75, -1.0]):

    # if signal > 0
    df['position'] = np.where(df.signal > 0,
                              # if signal greater than rolling median - 0.000001
                              np.where(df.signal > (df.signal.expanding(min_periods=1).quantile(0.50) - 0.000001),
                                       # return 1 if True, 0.75 if False
                                       posBins[0], posBins[1]),
                              # if signal <0 return -1
                              posBins[2])
    return df


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# run main

if __name__ == '__main__':

    # custom pandas settings
    setPandas()

    # set numpy float format
    floatFormatter = "{:,.6f}".format
    np.set_printoptions(formatter = {'float_kind':floatFormatter})

    path = '.\\_source\\'

    # # set logging
    # setLogging(LOGGING_DIRECTORY = os.path.join(path, 'logs', dt.today().strftime('%Y-%m-%d')),
    #            LOG_FILE_NAME = LOG_FILE_NAME,
    #            level = 'INFO')


    # --------------------------------------------------------------------------------------------------
    #
    try:

        # load SMA activity data
        dfSpy = fnLoadActivityFeed(ticker='SPY')
        dfFutures = fnLoadActivityFeed(ticker='ES_F')

        # aggregate activity feed data
        dfAgg = fnAggActivityFeed(dfSpy, dfFutures)

        nTrain=464
        nTest=len(dfAgg)-nTrain


        # nTrain = len(dfAgg.loc[dfAgg.index<='2018-01-01'])
        # nTest = len(dfAgg) - nTrain


        # --------------------------------------------------------------------------------------------------
        # make predictions (y)

        predictionsY = [predict(dfAgg[i:nTrain + nTest + i], nTrain, nTest) for i in range(0, len(dfAgg) - nTrain, nTest)]

        # for i in range(0, len(dfAgg)-nTrain,nTest):
            # print(dfAgg[i:nTrain + nTest + i])

        predictionsY = pd.DataFrame(predictionsY[0][1].T)

        predictionsY.index = dfAgg[nTrain:].index
        predictionsY.index.name='date'
        predictionsY.columns=['signal']
        predictionsY.to_csv('pred_y_rf_daily.csv', sep = ',')


        # --------------------------------------------------------------------------------------------------
        # calculate position based of predictions

        # todo:
        # make this adjustable
        dfAgg = dfAgg[598 - 450 - 100 + 1:]
        nTrain = 450
        nTest = 100

        # todo:
        # make this adjustable
        testIndex = dfAgg.loc[dfAgg.index >= '2018-01-02'].index

        dfPos = pd.DataFrame(data = [], index = testIndex, columns = ['position'])
        # positionsY= [fnCreatePositions(predict(dfAgg[i:nTrain+nTest+i],nTrain,nTest),dfPos) for i in range(0, len(dfAgg))]

        positionsY = [
                fnCreatePositions(predict(dfAgg[i:nTrain + nTest + i], nTrain, nTest), positionRecord = dfPos)
                for i in range(0, len(dfAgg) - nTrain - nTest, 1)
        ]

        positionsY = pd.DataFrame(positionsY, columns=['date','position'])
        positionsY.set_index('date', inplace=True)

        # positionsY.to_csv('pos_y.csv')
        # compute portfolio returns using position bins
        dfP = fnComputePortfolioRtn(predictionsY, positionsY)

        dfP.to_csv('backtest_results.csv')
















    # def plot(df,sellpoint,buypoint,start,last):
    #    newdf=copy.deepcopy(df)
    #    l1,=plt.plot(newdf['Close'][start:last],linewidth=1)
    #    plt.legend(handles=[l1],labels=['Close Price'],loc='best')
    #    for i in sellpoint:
    #        plt.plot(newdf.iloc[i,:].name,newdf['Close'][i],'rs',markersize=1.5)
    #        plt.annotate(str(newdf['Close'][i])[:5],xy=(newdf.iloc[i,:].name,newdf['Close'][i]),xytext=(newdf.iloc[i,:].name,newdf['Close'][i]+0.5),weight='ultralight')
    #    for i in buypoint:
    #        plt.plot(newdf.iloc[i,:].name,newdf['Close'][i],'ks',markersize=1.5)
    #        plt.annotate(str(newdf['Close'][i])[:5],xy=(newdf.iloc[i,:].name,newdf['Close'][i]),xytext=(newdf.iloc[i,:].name,newdf['Close'][i]+0.5),weight='ultralight')
    #    plt.show()
    #
    # plot(df,sellpoint,buypoint,250,1000)


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
