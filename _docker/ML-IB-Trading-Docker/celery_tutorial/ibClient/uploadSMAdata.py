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
# todo:
# parse CBOE futures term
# http://www.cboe.com/delayedquote/futures-quotes
# VIN, OVX, SKEW, USO

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
from celery_tutorial.fnLibrary import setPandas, fnUploadSQL, setOutputFilePath, setLogging, fnOdbcConnect

LOG_FILE_NAME = os.path.basename(__file__)


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# download VIX data from CBOE
# https://markets.cboe.com/us/futures/market_statistics/historical_data/products/csv/VX/2015-01-21/

def fnGetCBOEData(ticker='VIX', startDate=None, endDate=None):

    # download latest data from CBOE
    if ticker == 'VIX':
        url = 'https://ww2.cboe.com/publish/scheduledtask/mktdata/datahouse/{}current.csv'.format(ticker.lower())
        df = pd.read_csv(url, skiprows=1, index_col=0, parse_dates = True)

        df.rename(columns={'VIX Close':'VIX_Close'}, inplace=True)
        df = df[['VIX_Close']]

        dfV = fnComputeReturns(df, 'VIX_Close', tPeriod = 3, retType = 'log')
        dfV = pd.merge(dfV, fnComputeReturns(df, 'VIX_Close', tPeriod = 6, retType = 'log'), left_index=True, right_index = True)

        dfV = dfV.add_prefix('vix-')
        df = df.merge(dfV,left_index=True,right_index=True)

    else:
        url = 'https://ww2.cboe.com/publish/scheduledtask/mktdata/datahouse/{}dailyprices.csv'.format(ticker.lower())
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
    url = 'https://ww2.cboe.com/publish/scheduledtask/mktdata/datahouse/vixcurrent.csv'
    df = pd.read_csv(url, skiprows=1, index_col=0, parse_dates = True)

    df.rename(columns={'VIX Close':'VIX'}, inplace=True)
    df = df[['VIX']]

    # get 9-day vix
    url = 'https://ww2.cboe.com/publish/scheduledtask/mktdata/datahouse/vix9ddailyprices.csv'
    df9D = pd.read_csv(url, skiprows=3, index_col=0, parse_dates = True)['Close']
    df9D.index = pd.DatetimeIndex(df9D.index.str.replace('/', '-').str.replace('*', ''))

    # get 3 month VIX
    url = 'https://ww2.cboe.com/publish/scheduledtask/mktdata/datahouse/vix3mdailyprices.csv'
    df3M = pd.read_csv(url, skiprows=2, index_col=0, parse_dates = True)['CLOSE']
    df3M.index = pd.DatetimeIndex(df3M.index)

    df['VIX9D'] = df9D
    df['VIX3M'] = df3M
    df.dropna(inplace=True)

    if startDate:
        df = df.loc[df.index >= startDate]
    if endDate:
        df = df.loc[df.index <= endDate]

    df['VIX-VIX3M'] = df['VIX3M'] - df['VIX']
    df['VIX-VIX9D'] = df['VIX9D'] - df['VIX']
    df['VIX9D-VIX3M'] = df['VIX3M'] - df['VIX9D']

    # normalize and take log
    df['Norm_VIX-VIX3M'] =  np.where(df['VIX-VIX3M']<0, np.log(abs(df['VIX-VIX3M'])) * -1, np.log(abs(df['VIX-VIX3M'])) *1)
    df['Norm_VIX-VIX9D'] =  np.where(df['VIX-VIX9D']<0, np.log(abs(df['VIX-VIX9D'])) * -1, np.log(abs(df['VIX-VIX9D'])) * 1)
    df['Norm_VIX9D-VIX3M'] =  np.where(df['VIX9D-VIX3M']<0, np.log(abs(df['VIX9D-VIX3M'])) * -1, np.log(abs(df['VIX9D-VIX3M'])) * 1)

    df = df[df.columns[-3:]]

    return df


# --------------------------------------------------------------------------------------------------
# compute VIX Predictor
# defined as: VixAlert = 0 or 1. if the vix is more than 18% higher than two days before the first of the current month and also greater than 21 than set VixAlert=1 else it's always 0.

def fnVIXPredictor(df):
    alert = 0
    df['vix-ret'] = df['VIX_Close'].pct_change()
    df['month'] = df.index - pd.offsets.MonthBegin(1, normalize=True) - pd.DateOffset(days=2, normalize=True)
    return


# --------------------------------------------------------------------------------------------------
# calculate VIX futures contango

def fnVIXFuturesContango(days=2000):

    url = "http://vixcentral.com/historical/?days=%s" % days
    df = pd.read_html(url, flavor='bs4', header=0, index_col=0, parse_dates=True)[0]
    df = df[:-1]
    df.sort_index(ascending = True, inplace = True)

    # convert string percentage types to float
    cols = df.columns.to_list()[-3:]
    for i in cols:
        df[i] = df[i].str.strip("%").astype(float) / 100

    cols = df.columns.to_list()[:-3]
    for i in cols:
        df[i] = df[i].replace('-', 0).astype(float)

    df['warning'] = 0
    df['warning'] = np.where(df['Contango 2/1'] < -0.05, 1, 0)
    # df.loc[df['Contango 2/1']< -0.05, 'warning']=1

    df['warningSevere'] = 0
    df['warningSevere'] = np.where(df['Contango 2/1'] < -0.09, 1, 0)
    # df.loc[df['Contango 2/1']< -0.09, 'warningSevere']=1

    cols = df.columns.to_list()[-5:-2]
    for i in cols:
        # log difference to make stationary
        df[i] = np.log(abs(df[i])).diff()

    df.dropna(inplace=True)
    cols = df.columns.to_list()[-5:]
    df = df[cols]

    # replace 0% contango days (np.infinity)
    df = df.replace([np.inf, -np.inf], 0)
    return df


# --------------------------------------------------------------------------------------------------
# compute simple or log returns

def fnComputeReturns(df, colPrc='adjClose', tPeriod = None, retType = 'simple'):

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

def fnClassifyReturns(df, retType = 'simple'):

    df['rtnStdDev'] = df['{}-rtn-1D'.format(retType)].iloc[::1].rolling(30).std().iloc[::1]
    df['rtnStdDev'].dropna(inplace=True)
    # df['rtnStdDev'] = df['rtnStdDev'][1:]
    df.dropna(inplace=True)
    return df


# --------------------------------------------------------------------------------------------------
# calculate returns today / tomorrow and bin them

def fnCalculateLaggedRets(df, dfA):

    # compute returns
    dfR = fnComputeReturns(df, 'adjClose', tPeriod = 1, retType = 'log')

    # compute lagged returns
    dfR = pd.merge(dfR, fnComputeReturns(df, 'adjClose', tPeriod = 2, retType = 'log'), left_index = True, right_index = True)
    dfR = pd.merge(dfR, fnComputeReturns(df, 'adjClose', tPeriod = 3, retType = 'log'), left_index = True, right_index = True)
    dfR = pd.merge(dfR, fnComputeReturns(df, 'adjClose', tPeriod = 4, retType = 'log'), left_index = True, right_index = True)
    dfR = pd.merge(dfR, fnComputeReturns(df, 'adjClose', tPeriod = 5, retType = 'log'), left_index = True, right_index = True)

    df = df.merge(dfR, left_index = True, right_index = True)

    # compute moving averages and rolling z-score
    # df['adjClose_log'] = np.log(df.adjClose)
    # df['adjClose_Diff'] = df.adjClose_log - df.adjClose_log.shift(1)

    df['adjClose_Diff'] = np.log(df.adjClose).diff()

    window = 49
    df['ma-49D'] = SMA(df['adjClose_Diff'], timeperiod = window)
    colMean = df["ma-49D"]                                              # .rolling(window = window).mean()
    colStd = df["ma-49D"].rolling(window = window).std()
    df["ma-49D-zscore"] = (df['adjClose_Diff'] - colMean) / colStd

    # window = 194
    window = 99
    df['ma-{}D'.format(window)] = SMA(df['adjClose_Diff'], timeperiod = window)
    colMean = df["ma-{}D".format(window)]                                 # .rolling(window = window).mean()
    colStd = df["ma-{}D".format(window)].rolling(window = window).std()
    df["ma-{}D-zscore".format(window)] = (df['adjClose_Diff'] - colMean) / colStd

    # window = 309
    window = 149
    df['ma-{}D'.format(window)] = SMA(df['adjClose_Diff'], timeperiod = window)
    colMean = df["ma-{}D".format(window)]                                 # .rolling(window = window).mean()
    colStd = df["ma-{}D".format(window)].rolling(window = window).std()
    df["ma-{}D-zscore".format(window)] = (df['adjClose_Diff'] - colMean) / colStd

    # df = df.merge(dfR, left_index = True, right_index = True)

    # classify returns
    dfT = fnClassifyReturns(df, retType = 'log')

    # compute std deviation from simple returns
    rtnStdDev = dfT['log-rtn-1D'].iloc[::1].rolling(30).std().iloc[::1]
    rtnStdDev.dropna(inplace=True)

    dfT = dfT.loc[(dfT.index >= dfA.iloc[0].name) & (dfT.index <= dfA.iloc[-1].name)]
    rtnStdDev = rtnStdDev.loc[(rtnStdDev.index >= dfA.iloc[0].name) & (rtnStdDev.index <= dfA.iloc[-1].name)]

    ## using regular returns to calculate target variable
    rtnTodayToTomorrow = dfT['adjClose'].pct_change().shift(-1)

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

    colsDrop = ['adjClose']
    dfT.drop(columns = colsDrop, inplace = True)
    dfT.dropna(inplace = True)

    return dfT, rtnTodayToTomorrow, rtnTodayToTomorrowClassified


# --------------------------------------------------------------------------------------------------
# read in activity feed data

def fnLoadActivityFeed(ticker='SPY', startDate=None):

    if not startDate:
        startDate = '2015-01-02'
    # if not endDate:
    #     endDate = (dt.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    q = """
            SELECT * 
            FROM defaultdb.tblactivityfeedsma 
            WHERE ticker_tk = '%s' 
            AND timestampET >= '%s'
            ;   
        """ % (ticker, startDate)

    conn = fnOdbcConnect('defaultdb')

    df_temp = pd.read_sql_query(q, conn)
    conn.disconnect()
    conn.close()

    df_temp.sort_values('timestampET', inplace=True)
    dfAgg = df_temp.copy()

    # filtering based on trading hours and excluding weekends
    dfAgg = dfAgg.loc[(dfAgg['timestampET'].dt.dayofweek != 5) & (dfAgg['timestampET'].dt.dayofweek != 6)]
    dfAgg['Time'] = dfAgg['timestampET'].dt.strftime('%H:%M:%S')
    dfAgg['Date'] = dfAgg['timestampET'].dt.strftime('%Y-%m-%d')
    dfAgg = dfAgg[(dfAgg['Time'] >= '04:30:00') & (dfAgg['Time'] <= '16:00:00')]  # 69.75% cumulative ret

    # exclude weekends and drop empty columns
    dfAgg = dfAgg.dropna(axis = 'columns')
    dfAgg = dfAgg.drop(columns = ['ticker_tk', 'date', 'timestampET', 'description',
                                  'center-date', 'center-time', 'center-time-zone',
                                  'raw-s-delta', 'volume-delta'])

    dfAgg.rename(columns = { 'raw-s':'raw_s' }, inplace = True)

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

    dfT = dfT.drop(columns = ['ticker_at', 'Date', 'raw_s', 's-volume', 's-dispersion', 'Time', 'volume_base_s'])
    dfT.columns = ticker + ':' + dfT.columns

    return dfT


# --------------------------------------------------------------------------------------------------
# combine and aggregate spy / futures activity feed ata

def fnAggActivityFeed(df1, df2, dfStk, ticker=None):

    dfA = pd.concat([df1, df2], axis = 1, sort = False)

    # pull Spy returns, classified tommorrow returns, classified today returns
    df, rtnTodayToTomorrow, rtnTodayToTomorrowClassified = fnCalculateLaggedRets(dfStk, dfA)

    rtnTodayToTomorrow.index.name = 'Date'
    rtnTodayToTomorrowClassified.index.name = 'Date'

    dfA.dropna(inplace = True)
    rtnTodayToTomorrow.dropna(inplace = True)

    dfAgg = pd.merge(dfA, df, how = 'inner', left_index = True, right_index = True)
    dfAgg = pd.merge(dfAgg, rtnTodayToTomorrow, how = 'inner', left_index = True, right_index = True)
    dfAgg = pd.merge(dfAgg, rtnTodayToTomorrowClassified, how = 'inner', left_index = True, right_index = True)

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
    dfAgg.replace(False,0, inplace=True)
    dfAgg.replace(True,1, inplace=True)

    dfContango = fnVIXFuturesContango(days=2000)
    dfAgg = pd.merge(dfAgg, dfContango, how='left', left_index=True, right_index=True).fillna(method='ffill')

    dfAgg.drop(columns = [
            # 'Adj_Close_log',
            'adjClose_Diff',
            'Is_month_end', 'Is_month_start',
            'Is_quarter_end', 'Is_quarter_start',
            'Is_year_end', 'Is_year_start',
            # 'warning','warningSevere',
            # 'logRet-VIX_VIX3M',
            # 'logRet-VIX_VIX3M-2d',
            # 'ma-49D-zscore',
            # 'ma-149D-zscore',
            # 'Dayofweek',
    ],
               inplace = True)

    # to remove all sma features:
    # dfAgg=dfAgg.loc[:,dfAgg.columns[-26:]]

    return dfAgg


# --------------------------------------------------------------------------------------------------
# load security prices from defaultdb

def fnLoadTblSecurityPricesYahoo(ticker, startDate=None, endDate=None):

    if not startDate:
        startDate = '2015-01-02'
    if not endDate:
        endDate = (dt.today() - timedelta(days=1)).strftime('%Y-%m-%d')

    q = """
            SELECT * 
            FROM defaultdb.tblsecuritypricesyahoo 
            WHERE ticker_tk = '%s' 
            AND date >= '%s'
            AND date <= '%s'
            ;   
        """ % (ticker, startDate, endDate)

    conn = fnOdbcConnect('defaultdb')

    dfStk = pd.read_sql_query(q, conn)

    conn.disconnect()
    conn.close()

    dfStk['date'] = pd.to_datetime(dfStk['date'])
    dfStk.sort_values('date', inplace = True)
    dfStk.set_index('date', inplace = True)

    dfStk.drop(columns = ['ticker_at', 'ticker_tk',
                          'adjOpen', 'adjLow', 'adjHigh',
                          'volume', 'dividend', 'splitRatio'
                          ], inplace = True)

    return dfStk


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# run main
from celery_tutorial.celery import app

@app.task
def fnUploadSMA():

    # custom pandas settings
    setPandas()
    setLogging(LOGGING_DIRECTORY = '../logging/', LOG_FILE_NAME = LOG_FILE_NAME, level = LOG_LEVEL)


    # set numpy float format
    floatFormatter = "{:,.6f}".format
    np.set_printoptions(formatter = {'float_kind':floatFormatter})


    # --------------------------------------------------------------------------------------------------

    try:

        # load SMA activity data
        dfRaw = fnLoadActivityFeed(ticker = ticker)
        dfFutures = fnLoadActivityFeed(ticker = 'ES_F')

        # pull stk prices from table
        endDate = pd.to_datetime(datetime.now()).strftime('%Y-%m-%d')
        dfStk = fnLoadTblSecurityPricesYahoo(ticker, startDate = '2010-01-02', endDate = endDate)

        # aggregate feature data
        dfAgg = fnAggActivityFeed(dfRaw, dfFutures, dfStk = dfStk, ticker=ticker)

        dfLiveFeatures = dfAgg.copy()
        dfLiveFeatures['ticker_at'] = 'EQT'
        dfLiveFeatures['ticker_tk'] = ticker
        dfLiveFeatures.reset_index(inplace=True)
        dfLiveFeatures.rename(columns={'index':'date'},inplace=True)

        # upload live features
        conn = fnOdbcConnect('defaultdb')
        fnUploadSQL(dfLiveFeatures, conn, 'tbllivepredictionfeatures', 'REPLACE', None, True)
        conn.disconnect()
        conn.close()


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


# if __name__ == '__main__':
#
#     # custom pandas settings
#     setPandas()
#     setLogging(LOG_FILE_NAME = LOG_FILE_NAME, level = LOG_LEVEL)
#
#     path = '_source\\'
#
#     # set numpy float format
#     floatFormatter = "{:,.6f}".format
#     np.set_printoptions(formatter = {'float_kind':floatFormatter})
#
#
#     # --------------------------------------------------------------------------------------------------
#
#     try:
#
#         # load SMA activity data
#         dfRaw = fnLoadActivityFeed(ticker = ticker)
#         dfFutures = fnLoadActivityFeed(ticker = 'ES_F')
#
#         # pull stk prices from table
#         endDate = pd.to_datetime(datetime.now()).strftime('%Y-%m-%d')
#         dfStk = fnLoadTblSecurityPricesYahoo(ticker, startDate = '2010-01-02', endDate = endDate)
#
#         # aggregate feature data
#         dfAgg = fnAggActivityFeed(dfRaw, dfFutures, dfStk = dfStk, ticker=ticker)
#
#         dfLiveFeatures = dfAgg.copy()
#         dfLiveFeatures['ticker_at'] = 'EQT'
#         dfLiveFeatures['ticker_tk'] = ticker
#         dfLiveFeatures.reset_index(inplace=True)
#         dfLiveFeatures.rename(columns={'index':'date'},inplace=True)
#
#         # upload live features
#         conn = fnOdbcConnect('defaultdb')
#         fnUploadSQL(dfLiveFeatures, conn, 'tbllivepredictionfeatures', 'REPLACE', None, True)
#         conn.disconnect()
#         conn.close()
#
#
#         # --------------------------------------------------------------------------------------------------
#         # --------------------------------------------------------------------------------------------------
#         # close logger / handlers
#
#         logging.info("========== END PROGRAM ==========")
#
#     except Exception as e:
#         logging.error(str(e), exc_info=True)
#
#     # CLOSE LOGGING
#     for handler in logging.root.handlers:
#         handler.close()
#     logging.shutdown()
