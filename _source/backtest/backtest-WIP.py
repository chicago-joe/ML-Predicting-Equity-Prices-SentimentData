max_n = 251
wait_time = 0


# --------------------------------------------------------------------------------------------------
# backtest-WIP.py
#
#
# --------------------------------------------------------------------------------------------------
# created by Joseph Loss on 6/22/2020

# todo:
#


# --------------------------------------------------------------------------------------------------
# Module imports

import os, sys, logging
import copy
from pathlib import Path
from datetime import datetime as dt
import pandas as pd
import numpy as np
np.random.seed(seed=42)
import time
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
# --------------------------------------------------------------------------------------------------
# download VIX data from CBOE

def fnGetVIXData(startDate=None, endDate=None):

    # download latest data from CBOE
    url = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/vixcurrent.csv'

    df = pd.read_csv(url, skiprows=1, index_col=0, parse_dates = True)

    if startDate:
        df = df.loc[df.index >= startDate]
    if endDate:
        df = df.loc[df.index <= endDate]

    df = df[['VIX Close']]

    df = fnComputeReturns(df, 'VIX Close', retType = 'simple')
    df = fnComputeReturns(df, 'VIX Close', retType = 'log')

    df.rename(columns={'log-rtnYesterdayToToday':'VIX-log-rtnYesterdayToToday', 'log-rtnPriorTwoDays':'VIX-log-rtnPriorTwoDays',},inplace=True)

    return df


# --------------------------------------------------------------------------------------------------
# compute simple or log returns

def fnComputeReturns(df, colPrc='Adj_Close', retType = 'simple'):

    if retType.lower()=='log':
        df['{}-rtnYesterdayToToday'.format(retType)] = np.log(df[colPrc]).diff()
        df['{}-rtnPriorTwoDays'.format(retType)] = np.log(df[colPrc]).diff(2)

    elif retType.lower()=='simple':
        df['{}-rtnYesterdayToToday'.format(retType)] = df[colPrc].pct_change()
        df['{}-rtnPriorTwoDays'.format(retType)] = df[colPrc].pct_change(2)

    else:
        print('Please choose simple or log return type')

    return df


# --------------------------------------------------------------------------------------------------
# classify simple or log returns

# todo:
# Edit classification bins
def fnClassifyReturns(df, retType = 'simple'):

    df['rtnStdDev'] = df['{}-rtnYesterdayToToday'.format(retType)].iloc[::1].rolling(30).std().iloc[::1]
    df['rtnStdDev'].dropna(inplace=True)

    df['rtnStdDev'] = df['rtnStdDev'][1:]

    df.dropna(inplace=True)

    # classify returns TODAY based on std deviation * bin
    df['{}-rtnYesterdayToTodayClassified'.format(retType)] = [2 if df['{}-rtnYesterdayToToday'.format(retType)][date] > df['rtnStdDev'][date] * 1.0
                          else 1 if df['{}-rtnYesterdayToToday'.format(retType)][date] > df['rtnStdDev'][date] * 0.05
                          else -1 if df['{}-rtnYesterdayToToday'.format(retType)][date] < df['rtnStdDev'][date] * -0.05
                          else -2 if df['{}-rtnYesterdayToToday'.format(retType)][date] < df['rtnStdDev'][date] * -1.0
                          else 0 for date in df['rtnStdDev'].index]

    # df['{}-rtnPriorTwoDaysClassified'.format(retType)] = [2 if df['{}-rtnPriorTwoDays'.format(retType)][date] >= df['rtnStdDev'][date] * 1.0
    #                       else 1 if df['{}-rtnPriorTwoDays'.format(retType)][date] >= df['rtnStdDev'][date] * 0.05
    #                       else -1 if df['{}-rtnPriorTwoDays'.format(retType)][date] <= df['rtnStdDev'][date] * -0.05
    #                       else -2 if df['{}-rtnPriorTwoDays'.format(retType)][date] <= df['rtnStdDev'][date] * -1.0
    #                       else 0 for date in df['rtnStdDev'].index]

    return df


# --------------------------------------------------------------------------------------------------
# pull in SPY prices to calculate returns today / tomorrow and bin them

# noinspection DuplicatedCode
def fnLoadPriceData(ticker='SPY'):

    path = 'C:\\Users\\jloss\\PyCharmProjects\\ML-Predicting-Equity-Prices-SentimentData\\_source\\_data\\spyPrices\\'

    df = pd.read_csv(path + "SPY Price Data.csv")

    df.index = df['Date']
    df.index.name = 'date'
    df.sort_index(inplace = True)

    # compute returns
    df = fnComputeReturns(df, 'Adj_Close', retType = 'simple')
    dfR = fnComputeReturns(df, 'Adj_Close', retType = 'log')

    # classify returns
    dfC = fnClassifyReturns(dfR, retType = 'simple')
    dfT = fnClassifyReturns(dfC, retType = 'log')

    # compute std deviation from simple returns
    rtnStdDev = dfT['simple-rtnYesterdayToToday'].iloc[::1].rolling(30).std().iloc[::1]
    rtnStdDev.dropna(inplace=True)

    dfT = dfT.loc[(dfT.index>='2015-07-23') & (dfT.index<='2019-10-31')]

    rtnStdDev = rtnStdDev.loc[(rtnStdDev.index>='2015-07-23') & (rtnStdDev.index<='2019-10-31')]


    ## using regular returns to calculate target variable
    rtnTodayToTomorrow = dfT['Adj_Close'].pct_change().shift(-1)

    # classify returns TOMORROW based on std deviation * bin
    rtnTodayToTomorrowClassified = [
            2 if rtnTodayToTomorrow[date] > rtnStdDev[date] * 1.0
            else 1 if rtnTodayToTomorrow[date] > rtnStdDev[date] * 0.05
            else -1 if rtnTodayToTomorrow[date] < rtnStdDev[date] * -0.05
            else -2 if rtnTodayToTomorrow[date] < rtnStdDev[date] * -1.0
            else 0 for date in rtnStdDev.index]

    rtnTodayToTomorrow = pd.DataFrame(rtnTodayToTomorrow)
    rtnTodayToTomorrow.columns = ['rtnTodayToTomorrow']

    rtnTodayToTomorrowClassified = pd.DataFrame(rtnTodayToTomorrowClassified)
    rtnTodayToTomorrowClassified.index = rtnStdDev.index
    rtnTodayToTomorrowClassified.columns = ['rtnTodayToTomorrowClassified']


    colsDrop = ['Date', 'Open', 'High', 'Low', 'Close',
                'Volume', 'Dividend', 'Split', 'Adj_Open', 'Adj_High',
                'Adj_Low', 'Adj_Close', 'Adj_Volume', ]

    dfT.drop(columns = colsDrop, inplace = True)
    dfT.dropna(inplace = True)

    return dfT, rtnTodayToTomorrow, rtnTodayToTomorrowClassified


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

    # exclude weekends and drop empty columns
    dfAgg = dfAgg.dropna(axis = 'columns')
    dfAgg = dfAgg.drop(columns = ['ticker', 'date', 'description', 'center-date',
                            'center-time', 'center-time-zone', 'raw-s-delta', 'volume-delta'])

    # compute volume-base-s and ewm-volume-base-s
    dfAgg["volume_base_s"] = dfAgg["raw_s"] / dfAgg["s-volume"]
    dfAgg["ewm_volume_base_s"] = dfAgg.groupby("Date")["volume_base_s"].apply(lambda x:x.ewm(span = 30).mean())

    # aggregate by date
    dfT = dfAgg.groupby('Date').last().reset_index()
    dfT.index = dfT['Date']

    # compute factors
    dfT["mean_volume_base_s"] = dfAgg.groupby("Date")["volume_base_s"].mean()
    dfT["mean_raw_s"] = dfAgg.groupby("Date")["raw_s"].mean()
    dfT["mean_s_dispersion"] = dfAgg.groupby("Date")["s-dispersion"].mean()

    if ticker=='SPY':
        dfT['volume_base_s_z'] = (dfT['mean_volume_base_s'] - dfT['mean_volume_base_s'].rolling(14).mean()) \
                             / dfT['mean_volume_base_s'].rolling(14).std()
        dfT['s_dispersion_z'] = (dfT['mean_s_dispersion'] - dfT['mean_s_dispersion'].rolling(14).mean()) \
                            / dfT['mean_s_dispersion'].rolling(14).std()

    elif ticker == 'ES_F':
        dfT['volume_base_s_delta']=(dfT['mean_volume_base_s'][1:]-dfT['mean_volume_base_s'][:-1].values)
        dfT['s_dispersion_delta']=(dfT['mean_s_dispersion'][1:]-dfT['mean_s_dispersion'][:-1].values)

    dfT['raw_s_MACD_ewma6-ewma26'] = dfT["mean_raw_s"].ewm(span = 7).mean() - dfT["mean_raw_s"].ewm(span = 14).mean()

    dfT = dfT.drop(columns = ['Date', 'raw_s', 's-volume', 's-dispersion', 'Time', 'volume_base_s'])
    dfT.columns = ticker + ':' + dfT.columns

    return dfT


# --------------------------------------------------------------------------------------------------
# combine and aggregate spy / futures activity feed ata

def fnAggActivityFeed(df1, df2):

    dfA = pd.concat([df1, df2], axis = 1, sort = False)

    # pull Spy returns, classified tommorrow returns, classified today returns
    df, rtnTodayToTomorrow, rtnTodayToTomorrowClassified = fnLoadPriceData(ticker='SPY')

    rtnTodayToTomorrow.index.name = 'Date'
    rtnTodayToTomorrowClassified.index.name = 'Date'

    dfA.dropna(inplace = True)
    rtnTodayToTomorrow.dropna(inplace = True)

    dfAgg = pd.merge(dfA, df, how = 'inner', left_index = True, right_index = True)
    dfAgg = pd.merge(dfAgg, rtnTodayToTomorrow, how = 'inner', left_index = True, right_index = True)
    dfAgg = pd.merge(dfAgg, rtnTodayToTomorrowClassified, how = 'inner', left_index = True, right_index = True)

    # pull in VIX data
    dfVIX = fnGetVIXData(startDate = dfAgg.index[0], endDate = dfAgg.index[-1])

    dfVIX.rename(columns = { 'simple-rtnYesterdayToToday':'VIX-simple-rtnYesterdayToToday',
                             'simple-rtnPriorTwoDays':    'VIX-simple-rtnPriorTwoDays' }, inplace=True)
    dfVIX.dropna(inplace = True)

    # merge current features with VIX features
    dfAgg = pd.merge(dfAgg, dfVIX, how = 'inner', left_index = True, right_index = True)

    # drop collinear features
    dfAgg.drop(columns = [
            'simple-rtnYesterdayToToday',
            'simple-rtnPriorTwoDays',
            # 'log-rtnYesterdayToToday',
            # 'log-rtnPriorTwoDays',
            # 'log-rtnYesterdayToTodayClassified',
            # 'VIX Close',
            'simple-rtnYesterdayToTodayClassified',
            # 'VIX-simple-rtnYesterdayToToday',
            # 'VIX-simple-rtnPriorTwoDays',
            'VIX-log-rtnYesterdayToToday',
            'VIX-log-rtnPriorTwoDays'
    ],
            inplace = True)

    return dfAgg


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

    conditions = [(X_test < minTrain), (X_test > maxTrain)]
    choices = [minTrain, maxTrain]
    tmp = np.select(conditions, choices, default = X_test)

    X_test = pd.DataFrame._from_arrays(tmp.transpose(), columns = X_test.columns, index = X_test.index)


    # --------------------------------------------------------------------------------------------------
    # test for stationarity

    stationarityResults = (stationarity(X_train))

    stationarityResults = pd.DataFrame(stationarityResults, index = [0])
    stationaryFactors = []
    logFactors = []

    for i in stationarityResults.columns:
        if stationarityResults[i][0] == 1:
            stationaryFactors.append(i)

    cols = [
                'log-rtnYesterdayToTodayClassified',
                'ES_F:volume_base_s_delta',
                'ES_F:raw_s_MACD_ewma12-ewma26',
                'VIX-simple-rtnYesterdayToToday',
                'VIX-simple-rtnPriorTwoDays',
            ]

    X_test = X_test[stationaryFactors].drop(['rtnTodayToTomorrow',
                                             'rtnTodayToTomorrowClassified', ] + cols, axis = 1)
    X_train = X_train[stationaryFactors].drop(['rtnTodayToTomorrow',
                                               'rtnTodayToTomorrowClassified', ] + cols, axis = 1)

    # logging.debug('Final Stationary Factors:\n%s' % pd.Series(stationaryFactors))

    # X_train[logFactors]=np.log(X_train[logFactors]).diff().fillna(method='bfill').squeeze()
    # X_test[logFactors]=np.log(X_test[logFactors]).diff().fillna(method='bfill').squeeze()


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
                                    random_state = 42,
                                    n_jobs = -1)

    RFmodel.fit(X_train_std, y_train)

    temp = pd.Series(RFmodel.feature_importances_, index = X_test.columns)
    dfFeat[df.index[-1]] = temp

    # predict in-sample and out-of-sample
    y_train_pred = RFmodel.predict(X_train_std)
    y_test_pred = RFmodel.predict(X_test_std)


    # --------------------------------------------------------------------------------------------------
    # statistics

    print('\n\n----------------------------------------')
    print('DATE: %s' % pd.to_datetime(df.index[len(df)-1]).date())

    print('\nMSE \n Train: %.6f, Test: %.6f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))

    print('R^2 \n Train: %.4f, Test: %.4f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))

    print('Explained Variance \n Train: %.4f, Test: %.4f' % (
            explained_variance_score(y_train, y_train_pred),
            explained_variance_score(y_test, y_test_pred)))

    predictionsY = y_test_pred

    ## quantile is super common number so subtract .000001 or have quantile >= to the signal
    q1signal = np.quantile(predictionsY, 0.25) - 0.000001

    dfS.at[df.index[-1], 'q1signal'] = q1signal

    # dfS.at[df.index[-1], 'q1signalRolling'] = dfS['q1signal'].expanding(min_periods = 1).quantile(0.5) - 0.000001
    # dfS.at[df.index[-1], 'q1signalRolling'] = dfS['q1signal'].expanding(min_periods = 1).quantile(0.5)

    # dfS.at[df.index[-1], 'q1signalRolling'] = dfS['q1signal'].rolling(window = 15).quantile(0.25)

    # dfS.at[df.index[-1], 'q1signalRolling'] = dfS['q1signal'].rolling(window = 15).quantile(0.5)-0.000001
    # dfS.at[df.index[-1], 'q1signalRolling'] = dfS['q1signal'].rolling(window = 15).quantile(0.5)

    lastsignal = predictionsY[-1]

    dfS.at[df.index[-1], 'lastsignal'] = lastsignal

    dfS.at[df.index[-1], 'q1signalRolling'] = (dfS['lastsignal'] - 0.000001).rolling(window = 7).quantile(0.25).fillna(9999)



    # todo q1 signal < rolling signal = 0.75
    # if signal > 0

    if dfS.at[df.index[-1], 'q1signalRolling'] == 9999:
        dfS.at[df.index[-1], 'position'] = 0

    else:
        if (lastsignal > 0.0001) & (lastsignal > q1signal) & (lastsignal < 0.001):
            dfS.at[df.index[-1], 'position'] = 1.0
        elif 0.001 > lastsignal > 0.0001:
            dfS.at[df.index[-1], 'position'] = 0.75
        elif (lastsignal > 0.001) & (dfS.at[df.index[-1], 'q1signalRolling'] < lastsignal) & (dfS.at[df.index[-1], 'q1signal'] < dfS.at[df.index[-1], 'q1signalRolling']):
            dfS.at[df.index[-1], 'position'] = -1
        else:
            dfS.at[df.index[-1], 'position'] = 0


    print(' ')
    print('Last signal:\t%.6f' % dfS.at[df.index[-1], 'lastsignal'].astype(float))
    print('Q1 signal:\t\t %.6f' % dfS.at[df.index[-1], 'q1signal'].astype(float))
    print('Rolling signal:\t %.6f' % dfS.at[df.index[-1], 'q1signalRolling'].astype(float))
    print('\nCurrent Position: %.2f' % dfS.at[df.index[-1], 'position'].squeeze().astype(float))

    time.sleep(wait_time)

    featureImportances = pd.DataFrame(dfFeat)

    return [dfS, featureImportances]


# --------------------------------------------------------------------------------------------------
# compute portfolio returns with position sizing

def fnComputePortfolioRtn(pos):

    dfP = pos.copy()

    dfP.index = pd.DatetimeIndex(dfP.index.get_level_values(0))

    # set to time-aware index
    dfP.index = pd.DatetimeIndex(dfP.index)

    # import SPY returns
    path = 'C:\\Users\\jloss\\PyCharmProjects\\ML-Predicting-Equity-Prices-SentimentData\\_source\\_data\\spyPrices\\'

    dfU = pd.read_csv(path + "SPY Price Data.csv")

    dfU.index = dfU['Date']
    dfU.sort_index(inplace=True)

    dfU['rtnSPY'] = dfU['Adj_Close'].pct_change().shift(-1)
    dfU.index = pd.DatetimeIndex(dfU.index)
    dfP = dfP.merge(dfU[['Adj_Close','rtnSPY']], how = 'left', left_index = True, right_index = True)

    # calculate cumulative asset return
    dfP['cRtn-SPY'] = ((1 + dfP['rtnSPY']).cumprod() - 1).fillna(0)

    # calculate cumulative portfolio return
    dfP['rtnPort'] = dfP['position'] * dfP['rtnSPY']
    dfP['cRtn-Port'] = (1 + dfP['rtnPort']).cumprod()-1

    # todo:
    # xlsx formatting here

    # cols = ['signal',
    #         'Adj_Close',
    #         'rtnSPY',
    #         'cRtn-SPY',
    #         'position',
    #         'rtnPort',
    #         'cRtn-Port']
    #
    # dfP = dfP[cols]


    print('\n\n----------------------------------------')
    print('----------------------------------------')
    print('Starting Portfolio:\n%s' % dfP.head(5))
    print('\n')
    print('Ending Portfolio:\n%s' % dfP.tail(5))

    dfP.to_csv('backtest_results.csv')

    return dfP


# --------------------------------------------------------------------------------------------------
#  position binning

def fnPositionBinning(df, posBins=[1.0, 0.75, -1.0]):
    # todo:
    # tweak for one row
    # replace RiskToSPY with this function

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
# plot feature importances and pred vs residual values

def fnPlotFeatureImportance(model, dfFeat):

    dfFI = dfFeat.mean(axis=1)
    # plot Feature Importance of RandomForests model
    names = dfFI.index.tolist()
    featNames = np.array(names)

    featureImportance = dfFI.values
    featureImportance = featureImportance / featureImportance.max()    # scale by max importance
    sorted_idx = np.argsort(featureImportance)
    barPos = np.arange(sorted_idx.shape[0]) + 0.5
    plot.barh(barPos, featureImportance[sorted_idx], align = 'center')      # chart formatting
    plot.yticks(barPos, featNames[sorted_idx])
    plot.xlabel('Variable Importance')
    plot.show()

    return


    # --------------------------------------------------------------------------------------------------
    # plot predicted vs residual
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


    # --------------------------------------------------------------------------------------------------

    try:

        # load SMA activity data
        dfSpy = fnLoadActivityFeed(ticker='SPY')
        dfFutures = fnLoadActivityFeed(ticker='ES_F')

        # aggregate activity feed data
        dfAgg = fnAggActivityFeed(dfSpy, dfFutures)


        # --------------------------------------------------------------------------------------------------
        # calculate predictions based on rolling model (refit rolling 100 days)

        dfAgg = dfAgg[598 - 450 - 100 + 1:]

        nTrain = 440
        nTest = 100

        dfS = pd.DataFrame(index = [dfAgg[539:].index], columns = ['q1signal', 'q1signalRolling', 'lastsignal', 'position'])

        dpred = { }
        dfFeat = { }
        for i in range(0, len(dfAgg) - nTrain - nTest, 1):
            dpred, dfFeat = predict(dfAgg[i:nTrain + nTest + i], nTrain, nTest, dfS)

        dpred = dpred.iloc[:-1]
        print('\n')


        # --------------------------------------------------------------------------------------------------
        # compute portfolio returns using position bins

        dfP = fnComputePortfolioRtn(dpred)




        # todo:
        # plot cRet portfolio vs cRet SPY







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
