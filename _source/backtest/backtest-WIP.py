# --------------------------------------------------------------------------------------------------
# backtest-WIP.py
#
#
# --------------------------------------------------------------------------------------------------
# created by Joseph Loss on 6/22/2020

max_n = 251
seed = 42
wait_time = 0

# --------------------------------------------------------------------------------------------------
# todo:
# Also the 1 day, 2 day, 3 day log returns of VIX-VIX3M will probably help (as well as similar of VIX9D-VIX and VIX9D-VIX3m). not necessarily all of those but some
# VIN, OVX, SKEW, USO
# 2. put-call ratio
# 3. SMA cboe index http://www.cboe.com/index/dashboard/smlcw#smlcw-overview

# todo:
# http://www.cboe.com/products/vix-index-volatility/volatility-indexes
# 4. VIN and OVX, USO
# 6. Further time-frame tuning


# --------------------------------------------------------------------------------------------------
# Module imports

import os, logging, time
import pandas as pd
import numpy as np
np.random.seed(seed=seed)

from talib import SMA
import scipy.stats as stats
from scipy.stats import boxcox
from fastai.imports import *
from fastai.tabular.all import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from scipy.stats.mstats import winsorize
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import pylab as plot
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot as qp
import warnings
warnings.simplefilter("ignore")

# custom imports
from fnCommon import setPandas
LOG_FILE_NAME = os.path.basename(__file__)


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# download VIX data from CBOE

def fnGetCBOEData(ticker='VIX', startDate=None, endDate=None):

    # download latest data from CBOE
    if ticker == 'VIX':
        url = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/{}current.csv'.format(ticker.lower())
        df = pd.read_csv(url, skiprows=1, index_col=0, parse_dates = True)

        df.rename(columns={'VIX Close':'VIX_Close'}, inplace=True)
        df = df[['VIX_Close']]

        dfV = fnComputeReturns(df, 'VIX_Close', tPeriod = 3, retType = 'log')
        # dfV = pd.merge(dfV, fnComputeReturns(df, 'VIX_Close', tPeriod = 5, retType = 'log'), left_index=True,right_index = True)
        dfV = pd.merge(dfV, fnComputeReturns(df, 'VIX_Close', tPeriod = 6, retType = 'log'), left_index=True, right_index = True)

        dfV = dfV.add_prefix('vix-')
        df = df.merge(dfV,left_index=True,right_index=True)

    else:
        url = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/{}dailyprices.csv'.format(ticker.lower())
        df = pd.read_csv(url, skiprows=3, index_col=0, parse_dates = True)

    if startDate:
        df = df.loc[df.index >= startDate]
    if endDate:
        df = df.loc[df.index <= endDate]

    return df


# --------------------------------------------------------------------------------------------------
# compute VIX Term structure

def fnComputeVIXTerm(startDate=None, endDate=None):

    # download latest data from CBOE
    url = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/vixcurrent.csv'
    df = pd.read_csv(url, skiprows=1, index_col=0, parse_dates = True)

    df.rename(columns={'VIX Close':'VIX'}, inplace=True)
    df = df[['VIX']]

    # get 9-day vix
    url = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/vix9ddailyprices.csv'
    df9D = pd.read_csv(url, skiprows=3, index_col=0, parse_dates = True)['Close']
    df9D.index = pd.DatetimeIndex(df9D.index.str.replace('/', '-').str.replace('*', ''))

    # get 3 month VIX
    url = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/vix3mdailyprices.csv'
    df3M = pd.read_csv(url, skiprows=2, index_col=0, parse_dates = True)['CLOSE']
    df3M.index = pd.DatetimeIndex(df3M.index)

    df['VIX9D'] = df9D
    df['VIX3M'] = df3M
    df.dropna(inplace=True)

    if startDate:
        df = df.loc[df.index >= startDate]
    if endDate:
        df = df.loc[df.index <= endDate]

    # compute VIX Term Spreads
    df['VIX-VIX3M'] = abs(df['VIX'] - df['VIX3M'])
    df['VIX-VIX9D'] = abs(df['VIX'] - df['VIX9D'])
    df['VIX9D-VIX3M'] = abs(df['VIX9D'] - df['VIX3M'])

    dfVR = fnComputeReturns(df, 'VIX-VIX3M', tPeriod=1, retType='log')
    dfVR = pd.merge(dfVR, fnComputeReturns(df, 'VIX-VIX9D', tPeriod = 1, retType = 'log'), left_index = True, right_index = True)
    dfVR = pd.merge(dfVR, fnComputeReturns(df, 'VIX9D-VIX3M', tPeriod = 1, retType = 'log'), left_index = True, right_index = True)
    dfVR = pd.merge(dfVR, fnComputeReturns(df, 'VIX-VIX3M', tPeriod = 2, retType = 'log'), left_index = True, right_index = True)
    dfVR = pd.merge(dfVR, fnComputeReturns(df, 'VIX-VIX3M', tPeriod = 3, retType = 'log'), left_index = True, right_index = True)

    cols = ['logRet-VIX_VIX3M', 'logRet-VIX_VIX9D', 'logRet-VIX9D_VIX3M','logRet-VIX_VIX3M-2d', 'logRet-VIX_VIX3M-3d']
    dfVR.columns = cols
    dfVR.dropna(inplace=True)

    return dfVR


# --------------------------------------------------------------------------------------------------
# compute VIX Predictor

# defined as: VixAlert = 0 or 1. if the vix is more than 18% higher than two days before the first of the current month and also greater than 21 than set VixAlert=1 else it's always 0.

def fnVIXPredictor(df):
    alert = 0
    df['vix-ret'] = df['VIX_Close'].pct_change()
    df['month'] = df.index - pd.offsets.MonthBegin(1,normalize=True) - pd.DateOffset(days=2,normalize=True)





# --------------------------------------------------------------------------------------------------
# compute simple or log returns

def fnComputeReturns(df, colPrc='Adj_Close', tPeriod = None, retType = 'simple'):

    if retType.lower()=='log':
        ret = pd.Series(np.log(df[colPrc]).diff(tPeriod))
        ret.name = '{}-rtn-{}D'.format(retType, tPeriod)

    elif retType.lower()=='simple':
        ret = pd.Series(df[colPrc].pct_change(tPeriod))
        ret.name = '{}-rtn-{}D'.format(retType, tPeriod)

    else:
        print('Please choose simple or log return type')

    return ret


# --------------------------------------------------------------------------------------------------
# classify simple or log returns

# todo:
# Edit classification bins
def fnClassifyReturns(df, retType = 'simple'):

    df['rtnStdDev'] = df['{}-rtn-1D'.format(retType)].iloc[::1].rolling(30).std().iloc[::1]
    df['rtnStdDev'].dropna(inplace=True)
    # df['rtnStdDev'] = df['rtnStdDev'][1:]
    df.dropna(inplace=True)

    # --------------------------------------------------------------------------------------------------
    # classify returns TODAY based on std deviation * bin

    # df.loc[df['{}-rtn-1D'.format(retType)] > (df['rtnStdDev'] * 1.0),
    # '{}-rtnYesterdayToTodayClassified'.format(retType)] = 2

    # df.loc[(df['{}-rtn-1D'.format(retType)] > (df['rtnStdDev'] * 0.05)) & (df['{}-rtnYesterdayToTodayClassified'.format(retType)].isna()),
    # '{}-rtnYesterdayToTodayClassified'.format(retType)] = 1
    # df.loc[(df['{}-rtn-1D'.format(retType)] < (df['rtnStdDev'] * -0.05)) & (df['{}-rtnYesterdayToTodayClassified'.format(retType)].isna()),
    # '{}-rtnYesterdayToTodayClassified'.format(retType)] = -1

    # df.loc[(df['{}-rtn-1D'.format(retType)] < (df['rtnStdDev'] * -1.0)) & (df['{}-rtnYesterdayToTodayClassified'.format(retType)].isna()),
    # '{}-rtnYesterdayToTodayClassified'.format(retType)] = -2

    # df.loc[df['{}-rtnYesterdayToTodayClassified'.format(retType)].isna(),
    # '{}-rtnYesterdayToTodayClassified'.format(retType)] = 0

    ## df['log-rtn-1DClassified'].value_counts()
    # -1.00000    1112
    # 1.00000      944
    # 2.00000      443
    # 0.00000      159

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
    dfR = fnComputeReturns(df, 'Adj_Close', tPeriod = 1, retType = 'log')

    # compute lagged returns
    dfR = pd.merge(dfR, fnComputeReturns(df, 'Adj_Close', tPeriod = 2, retType = 'log'), left_index = True, right_index = True)
    dfR = pd.merge(dfR, fnComputeReturns(df, 'Adj_Close', tPeriod = 3, retType = 'log'), left_index = True, right_index = True)
    dfR = pd.merge(dfR, fnComputeReturns(df, 'Adj_Close', tPeriod = 4, retType = 'log'), left_index = True, right_index = True)
    dfR = pd.merge(dfR, fnComputeReturns(df, 'Adj_Close', tPeriod = 5, retType = 'log'), left_index = True, right_index = True)

    df = df.merge(dfR, left_index = True, right_index = True)

    # compute moving averages and rolling z-score
    df['Adj_Close_log'] = np.log(df.Adj_Close)
    df['Adj_Close_Diff'] = df.Adj_Close_log - df.Adj_Close_log.shift(1)

    window = 49
    df['ma-49D'] = SMA(df['Adj_Close_Diff'], timeperiod = window)
    colMean = df["ma-49D"]                                  # .rolling(window = window).mean()
    colStd = df["ma-49D"].rolling(window = window).std()
    df["ma-49D-zscore"] = (df['Adj_Close_Diff'] - colMean) / colStd

    # window = 194
    window = 99
    df['ma-{}D'.format(window)] = SMA(df['Adj_Close_Diff'], timeperiod = window)
    colMean = df["ma-{}D".format(window)]                                 # .rolling(window = window).mean()
    colStd = df["ma-{}D".format(window)].rolling(window = window).std()
    df["ma-{}D-zscore".format(window)] = (df['Adj_Close_Diff'] - colMean) / colStd

    # window = 309
    window = 149
    df['ma-{}D'.format(window)] = SMA(df['Adj_Close_Diff'], timeperiod = window)
    colMean = df["ma-{}D".format(window)]                                 # .rolling(window = window).mean()
    colStd = df["ma-{}D".format(window)].rolling(window = window).std()
    df["ma-{}D-zscore".format(window)] = (df['Adj_Close_Diff'] - colMean) / colStd

    # df = df.merge(dfR, left_index = True, right_index = True)

    # classify returns
    dfT = fnClassifyReturns(df, retType = 'log')

    # compute std deviation from simple returns
    rtnStdDev = dfT['log-rtn-1D'].iloc[::1].rolling(30).std().iloc[::1]
    rtnStdDev.dropna(inplace=True)


    # todo:
    #  adjust these automatically

    dfT = dfT.loc[(dfT.index >= '2015-07-23') & (dfT.index <= '2019-10-31')]
    rtnStdDev = rtnStdDev.loc[(rtnStdDev.index >= '2015-07-23') & (rtnStdDev.index <= '2019-10-31')]

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

    dfT['raw_s_MACD_ewma7-ewma14'] = dfT["mean_raw_s"].ewm(span = 7).mean() - dfT["mean_raw_s"].ewm(span = 14).mean()

    dfT = dfT.drop(columns = ['Date', 'raw_s', 's-volume', 's-dispersion', 'Time', 'volume_base_s'])
    dfT.columns = ticker + ':' + dfT.columns

    return dfT


# --------------------------------------------------------------------------------------------------
# read in S-Factor Feed Data

def fnLoadSFactorFeed(ticker='SPY'):

    path = 'C:\\Users\\jloss\\PyCharmProjects\\ML-Predicting-Equity-Prices-SentimentData\\_source\\_data\\sFactorFeed\\'

    colNames = ['ticker', 'date', 'raw-s', 'raw-s-mean', 'raw-volatility',
                'raw-score', 's', 's-mean', 's-volatility', 's-score',
                's-volume', 'sv-mean', 'sv-volatility', 'sv-score',
                's-dispersion', 's-buzz', 's-delta',
                'center-date', 'center-time', 'center-time-zone']

    dfSpy2015 = pd.read_csv(path + '{}2015.txt'.format(ticker), skiprows = 4, sep = '\t')
    dfSpy2016 = pd.read_csv(path + '{}2016.txt'.format(ticker), skiprows = 4, sep = '\t')
    dfSpy2017 = pd.read_csv(path + '{}2017.txt'.format(ticker), skiprows = 4, sep = '\t')
    dfSpy2018 = pd.read_csv(path + '{}2018.txt'.format(ticker), skiprows = 4, sep = '\t')
    dfSpy2019 = pd.read_csv(path + '{}2019.txt'.format(ticker), skiprows = 4, sep = '\t')

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
    dfAgg = dfAgg.drop(columns = ['ticker', 'date', 'raw-s', 'raw-s-mean', 'raw-volatility', 'raw-score',
                                  'center-date', 'center-time', 'center-time-zone'])

    # aggregate by date
    dfT = dfAgg.groupby('Date').last().reset_index()
    dfT.index = dfT['Date']

    dfT = dfT.drop(columns = ['Date', 'Time'])
    dfT.columns = ticker + ':' + dfT.columns

    return dfT


# --------------------------------------------------------------------------------------------------
# combine and aggregate spy / futures activity feed ata

def fnAggActivityFeed(df1, df2, df3, df4):

    df3.index = pd.to_datetime(df3.index).strftime('%Y-%m-%d')
    df4.index = pd.to_datetime(df4.index).strftime('%Y-%m-%d')

    dfA = pd.concat([df1, df2], axis = 1, sort = False)
    dfB = pd.concat([df3, df4], axis=1, sort=False)

    # pull Spy returns, classified tommorrow returns, classified today returns
    df, rtnTodayToTomorrow, rtnTodayToTomorrowClassified = fnLoadPriceData(ticker='SPY')

    rtnTodayToTomorrow.index.name = 'Date'
    rtnTodayToTomorrowClassified.index.name = 'Date'

    dfA.dropna(inplace = True)
    dfB.dropna(inplace=True)
    rtnTodayToTomorrow.dropna(inplace = True)

    dfAgg = pd.merge(dfA, df, how = 'inner', left_index = True, right_index = True)
    dfAgg = pd.merge(dfAgg, rtnTodayToTomorrow, how = 'inner', left_index = True, right_index = True)
    dfAgg = pd.merge(dfAgg, rtnTodayToTomorrowClassified, how = 'inner', left_index = True, right_index = True)

    # dfAgg = pd.merge(dfAgg, dfB, how='inner',left_index=True,right_index=True)

    # pull in VIX data
    dfVIX = fnGetCBOEData(ticker='VIX', startDate = dfAgg.index[0], endDate = dfAgg.index[-1])
    dfVIX.dropna(inplace = True)

    # merge current features with VIX features
    dfAgg = pd.merge(dfAgg, dfVIX, how = 'inner', left_index = True, right_index = True)

    dfVIXTerm = fnComputeVIXTerm(startDate = dfAgg.index[0], endDate = dfAgg.index[-1])
    dfAgg = pd.merge(dfAgg, dfVIXTerm, how = 'inner', left_index = True, right_index = True)

    dfAgg['date'] = dfAgg.index
    make_date(dfAgg,'date')
    test_eq(dfAgg['date'].dtype, np.dtype('datetime64[ns]'))

    dfAgg = add_datepart(dfAgg, 'date')
    dfAgg.drop(columns=['Elapsed'], inplace=True)
    dfAgg.replace(False,0,inplace=True)
    dfAgg.replace(True,1,inplace=True)

    # test_eq(dfAgg.columns[-13:], ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear', 'Is_month_end', 'Is_month_start',
    #         'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start'])

    dfAgg.drop(columns = [
            'Adj_Close_log', 'Adj_Close_Diff',
                          # 'Is_month_end', 'Is_month_start',
                          # 'Is_quarter_end', 'Is_quarter_start',
                          # 'Is_year_end','Is_year_start'
    ],
               inplace = True)

    # dfAgg.drop(columns=['ma-309D-zscore', 'ma-194D-zscore', 'SPY:volume_base_s_z',
    #    'ES_F:s-volatility', 'ES_F:s-score', 'VIX_Close', 'ES_F:sv-score',
    #    'SPY:raw_s_MACD_ewma7-ewma14', 'SPY:s_dispersion_z', 'vix-log-rtn-6D',
    #    'ES_F:volume_base_s_delta', 'ES_F:ewm_volume_base_s', 'SPY:s-score',
    #    'ma-49D-zscore', 'Dayofweek'], inplace=True)


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

    for i in stationarityResults.columns:
        if stationarityResults[i][0] == 1:
            stationaryFactors.append(i)

    # colsAppend = ['ma-49D', 'ma-194D', 'ma-194D-zscore',
    #               'ma-309D', 'ma-309D-zscore',
    #               'Year', 'Month', 'Week', 'Dayofyear', 'Is_quarter_start', 'Is_year_end','Is_year_start']

    # stationaryFactors.extend(colsAppend)


    X_test = X_test[stationaryFactors].drop(['rtnTodayToTomorrow',
                                             'rtnTodayToTomorrowClassified', ]
                                            # + cols
                                            ,
                                            axis = 1)
    X_train = X_train[stationaryFactors].drop(['rtnTodayToTomorrow',
                                               'rtnTodayToTomorrowClassified', ]
                                              # + cols
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
    # dfFeat = temp.transpose()

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

    lastsignal = predictionsY[-1]
    dfS.at[df.index[-1], 'lastsignal'] = lastsignal
    dfS.at[df.index[-1], 'q1signalRolling'] = (dfS['lastsignal'] - 0.000001).rolling(window = 7).quantile(0.25).fillna(9999)


    # todo:
    #  q1 signal < rolling signal = 0.75
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


    # todo: xlsx formatting here
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
# plot feature importances and pred vs residual values

def fnPlotFeatureImportance(dfFeat):

    dfFI = dfFeat.mean(axis=1)
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

def plotResiduals():

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
    return


# --------------------------------------------------------------------------------------------------
# plot distributions and quantile plots

def plotDistributions(df):
    for i in dfAgg.columns:
        sns.distplot(dfAgg[i])
        plt.show()
        fig = qp(dfAgg[i], line='s')
        plt.show()
    return


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

        dfSpyFactor = fnLoadSFactorFeed(ticker='SPY')
        dfFuturesFactor = fnLoadSFactorFeed(ticker='ES_F')

        # aggregate activity feed data
        dfAgg = fnAggActivityFeed(dfSpy, dfFutures, dfSpyFactor, dfFuturesFactor)


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
        # compute portfolio returns

        dfP = fnComputePortfolioRtn(dpred)

        # plot feature importance
        print('Mean Feature Importance:\n', dfFeat.mean(axis = 1).sort_values(ascending = False))
        fnPlotFeatureImportance(dfFeat = dfFeat)


        # todo:
        # plot cRet portfolio vs cRet SPY





    # --------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------
    # close logger / handlers

        print("========== END PROGRAM ==========")
        logging.info("========== END PROGRAM ==========")

    except Exception as e:
        logging.error(str(e), exc_info=True)

    # CLOSE LOGGING
    for handler in logging.root.handlers:
        handler.close()
    logging.shutdown()
