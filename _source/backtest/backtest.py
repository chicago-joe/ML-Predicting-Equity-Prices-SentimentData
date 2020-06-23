################################################################
# ensembling.py
# Ensemble Methods
# Created by Joseph Loss on 11/06/2019
#
# Contact: loss2@illinois.edu
###############################################################
import pandas as pd
import numpy as np
import os, sys
from fnCommon import setPandas, setLogging, setOutputFile
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, accuracy_score, explained_variance_score
from pathlib import Path
# import pylab as plot
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from scipy.stats.mstats import winsorize
from statsmodels.tsa.stattools import adfuller
import warnings
warnings.simplefilter("ignore")
import copy



# --------------------------------------------------------------------------------------------------
#

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
        rtnClassified = 2
    elif rtn > (rtnStdDev * stdDevBins[1]):
        rtnClassified = 1
    elif rtn > (rtnStdDev * stdDevBins[2]):
        rtnClassified = 0
    elif rtn > (rtnStdDev * stdDevBins[3]):
        rtnClassified = -1
    else:
        rtnClassified = -2

    return rtnClassified









# --------------------------------------------------------------------------------------------------
# pull in SPY prices to calculate returns today / tomorrow and bin them

# noinspection DuplicatedCode
def SPY():
    path = 'C:\\Users\\jloss\\PyCharmProjects\\ML-Predicting-Equity-Prices-SentimentData\\_source\\_data\\spyPrices\\'

    dfSPY = pd.read_csv(path + "SPY.csv")
    dfSPY.index = dfSPY['Date']

    rtnTommorrow = (dfSPY['Adj_Close'][:-1].values - dfSPY['Adj_Close'][1:]) / dfSPY['Adj_Close'][1:]
    rtnToday = (dfSPY['Adj_Close'][:-1] - dfSPY['Adj_Close'][1:].values) / dfSPY['Adj_Close'][1:].values

    # type(rtnToday)

    rtnStdDev=rtnToday.iloc[::-1].rolling(250).std().iloc[::-1]
    rtnStdDev=rtnStdDev.dropna()
    rtnStdDev=rtnStdDev[1:]
    
    rtnTommorrowClassified = [2 if rtnTommorrow[date] > rtnStdDev[date] * 1.0 
                     else 1 if rtnTommorrow[date] > rtnStdDev[date] * 0.05 
                     else 0 if rtnTommorrow[date] > rtnStdDev[date] * -0.05 
                     else -1 if rtnTommorrow[date] > rtnStdDev[date] * -1.0 
                     else -2 for date in rtnStdDev.index]

    # rtnClassified = fnClassifyReturns(rtnBins=[2,1,0,-1,-2],stdDevBins = [1.0,0.05,-0.05,-1],rtn=rtnTommorrow, rtnStdDev = rtnStdDev)
    # rtnClassified=[ 2  if ret>s2 else 1 if ret>s1 else 0 if ret>s0 else -1 if ret>sn1 else -2 for ret in rtnTommorrow]


    dfTMC = pd.DataFrame(rtnTommorrowClassified)
    dfTMC.index = rtnStdDev.index
    dfTMC.columns = ['rtnClassified']

    rtnTodayClassified = [2 if rtnToday[date] > rtnStdDev[date] * 1.0 
                          else 1 if rtnToday[date] > rtnStdDev[date] * 0.05 
                          else 0 if rtnToday[date] > rtnStdDev[date] * -0.05 
                          else -1 if rtnToday[date] > rtnStdDev[date] * -1.0 
                          else -2 for date in rtnStdDev.index]

    # rtnClassified=[ 2  if ret>s2 else 1 if ret>s1 else 0 if ret>s0 else -1 if ret>sn1 else -2 for ret in rtnTommorrow]

    dfTOC = pd.DataFrame(rtnTodayClassified)
    dfTOC.index = rtnStdDev.index
    dfTOC.columns = ['rtnTodayClassified']

    rtnTommorrow = pd.DataFrame(rtnTommorrow)
    rtnToday = pd.DataFrame(rtnToday)
    rtnTommorrow.columns = ['rtnTommorrow']
    rtnToday.columns = ['rtnToday']

    return rtnToday, rtnTommorrow, dfTMC, dfTOC


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
    
    dfT['volume_base_s_z'] = (dfT['mean_volume_base_s'] - dfT['mean_volume_base_s'].rolling(26).mean()) \
                             / dfT['mean_volume_base_s'].rolling(26).std()
    dfT['s_dispersion_z'] = (dfT['mean_s_dispersion'] - dfT['mean_s_dispersion'].rolling(26).mean()) \
                            / dfT['mean_s_dispersion'].rolling(26).std()
    
    dfT['raw_s_MACD_ewma12-ewma26'] = dfT["mean_raw_s"].ewm(span = 12).mean() - dfT["mean_raw_s"].ewm(span = 26).mean()

    dfT = dfT.drop(columns = ['Date', 'raw_s', 's-volume', 's-dispersion', 'Time', 'volume_base_s'])
    dfT.columns = "spy_" + dfT.columns

    return dfT


    # --------------------------------------------------------------------------------------------------
    # combine and aggregate spy / futures activity feed ata

def fnAggActivityFeed(df1, df2):

    dfA = pd.concat([df1, df2], axis = 1, sort = False)

    # pull Spy returns, classified tommorrow returns, classified today returns
    rtnToday, rtnTommorrow, dfTMC, dfTOC = SPY()

    rtnStdDev = rtnToday.iloc[::-1].rolling(250).std().iloc[::-1]
    rtnStdDev = rtnStdDev.dropna()
    rtnStdDev = rtnStdDev[1:]

    rtnStdDev.columns = ['rtnStdDev']

    dfAgg = pd.concat([dfA, rtnTommorrow, rtnToday, dfTMC, rtnStdDev, dfTOC],
                      axis = 1,
                      sort = False,
                      join = 'inner').dropna()

    return dfAgg


# --------------------------------------------------------------------------------------------------
# winsorize data method

def using_mstats(s):
    return winsorize(s, limits = [0.005, 0.005])


# --------------------------------------------------------------------------------------------------
# adf testing

def adf_test(timeseries):
    # print('Results of Augment Dickey-Fuller Test:')

    dftest = adfuller(timeseries, autolag = 'AIC')
    dfoutput = pd.Series(dftest[0:4], index = ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    return (dfoutput)


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

def predict(df, ntrain, ntest):

    print(df.index[len(df) - 1])

    X_train = df[0:ntrain]
    X_test = df[ntrain:ntrain + ntest]

    y_train = X_train['next_Return']
    #    train_y_class=df_train['classret']
    y_test = X_test['next_Return']
    #    test_y_class=df_test['classret']

    X_train.drop(['next_Return', 'classret'], axis = 1)
    X_test.drop(['next_Return', 'classret'], axis = 1)

    X_train = X_train.apply(using_mstats, axis = 0)
    maxtrain = X_train.max()
    mintrain = X_train.min()

    for col in X_test:
        X_test[col][X_test[col] < mintrain[col]] = mintrain[col]
        X_test[col][X_test[col] > maxtrain[col]] = maxtrain[col]

    # test for stationarity
    slist = (stationarity(X_train))

    slist = pd.DataFrame(slist, index = [0])
    factors = []

    for i in slist.columns:
        if slist[i][0] == 1:
            factors.append(i)

    factors.remove("classret")
    factors.remove("next_Return")

    X_train = X_train[factors]
    X_test = X_test[factors]

    # Preprocess / Standardize data
    sc_X = StandardScaler()
    X_train_std = sc_X.fit_transform(X_train)
    X_test_std = sc_X.transform(X_test)

    y_train = np.array(y_train).reshape(-1, 1)
    y_test = np.array(y_test).reshape(-1, 1)

    best_leaf_nodes = 2
    best_n = 100

    RFmodel = RandomForestRegressor(n_estimators = best_n, max_leaf_nodes = best_leaf_nodes, n_jobs = -1)
    # RFmodel = RandomForestClassifier(n_estimators = best_n,max_leaf_nodes=best_leaf_nodes,n_jobs=-1)
    RFmodel.fit(X_train_std, y_train)

    # predict on in-sample and oos
    y_train_pred = RFmodel.predict(X_train_std)
    y_test_pred = RFmodel.predict(X_test_std)

    print([df.index[len(df) - 1], y_test_pred])
    print('MSE train: %.3f, test: %.3f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))

    print('R^2 train: %.3f, test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))

    # print('accuracy train: %.3f, test: %.3f' % (
    #        accuracy_score(y_train, y_train_pred),
    #        accuracy_score(y_test, y_test_pred)))

    print('explanined variance train: %.3f, test: %.3f' % (
            explained_variance_score(y_train, y_train_pred),
            explained_variance_score(y_test, y_test_pred)))
    return [df.index[len(df) - 1], y_test_pred]


# --------------------------------------------------------------------------------------------------
# determine the position based off pred_y

def position(pred_y):

    index_y = pred_y[0]
    pred_y = pred_y[1]

    f = open("tempsave.csv", "a")
    f.write("\n")

    q1signal = np.quantile(pred_y, 0.25) - 0.000001
    print('q1:', q1signal)
    lastsignal = pred_y[len(pred_y) - 1]
    print('pred', lastsignal)

    # --------------------------------------------------------------------------------------------------
    # determine risk to spy based of 1st quantile signal

    def risk(val, q1signal):
        if val > q1signal:
            return 1
        elif val > 0:
            return 0.75
        else:
            return 2

    riskToSpy = (len(pred_y) - 1) / sum([risk(n, q1signal) for n in pred_y])

    if (lastsignal > q1signal and lastsignal > 0.000001):
        f.write(index_y + ',' + str(riskToSpy))
        f.close()
        return [index_y, riskToSpy]

    elif (lastsignal > 0.000001):
        f.write(index_y + ',' + str(0.75 * riskToSpy))
        f.close()
        return [index_y, 0.75 * riskToSpy]

    elif (lastsignal > 0):
        f.write(index_y + ',' + str(0))
        f.close()
        return [index_y, 0]

    else:
        f.write(index_y + ',' + str(-1))
        f.close()
        return [index_y, -1]


if __name__ == '__main__':

    setPandas()

    dfSpy = fnLoadActivityFeed(ticker='SPY')
    dfFutures = fnLoadActivityFeed(ticker='ES_F')

    dfAgg = fnAggActivityFeed(dfSpy, dfFutures)

    ntrain = 464
    ntest = len(dfAgg) - ntrain

    pred_y = [predict(dfAgg[i:ntrain + ntest + i], ntrain, ntest) for i in range(0, len(dfAgg) - ntrain, ntest)]
    pred_y = pd.DataFrame(pred_y[0][1].T)

    pred_y.index = dfAgg[ntrain:].index
    pred_y.to_csv('pred_y_rf_daily.csv', sep = ',')

    dfAgg = dfAgg[598 - 450 - 100 + 1:]
    ntrain = 450
    ntest = 100

    pos_y = [position(predict(dfAgg[i:ntrain + ntest + i], ntrain, ntest)) for i in range(0, len(dfAgg) - ntrain - ntest, 1)]
    pd.DataFrame(pos_y).to_csv('pos_y_rf_daily.csv', sep = ',')




    print('----- END PROGRAM -----')










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
