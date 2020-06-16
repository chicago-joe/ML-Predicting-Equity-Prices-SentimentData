# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 17:36:04 2019

@author: duany
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 21:35:50 2019

@author: duany
"""
import pandas as pd
import time
import numpy as np
# import PyWavelets
# from PyWavelets import wavedec
# from PyWavelets import wt

import pywt
from pywt import wavedec
# from pywt import wt

from datetime import timedelta
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


# def acf_pacf_plot(ts_log_diff,name):
#     sm.graphics.tsa.plot_acf(ts_log_diff,lags=40) #ARIMA,q
#     plt.show()
#     plt.savefig('./station/'+name+'acf.png')
#     sm.graphics.tsa.plot_pacf(ts_log_diff,lags=40) #ARIMA,p
#     plt.show()
#     plt.savefig('./station/'+name+'pacf.png')


# def stationarity(result):
#     plist={}
#     for col in result:
#         acf_pacf_plot(result[col],col)
#         if adf_test(result[col])['p-value']<0.05:
#             st=True
#         else:
#             st=False
#         plist[col]=st
#     return plist
#
# def adf_test(timeseries):
#     print('Results of Augment Dickey-Fuller Test:')
    # dftest = adfuller(timeseries, autolag='AIC')
    # dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    # for key,value in dftest[4].items():
    #     dfoutput['Critical Value (%s)'%key] = value
    # return(dfoutput)

def SPY():
    spyprice=pd.read_csv('SPY Price Data.csv')
    spyprice.index=spyprice['Date']
    next_Return=(spyprice['Adj_Close'][:-1].values-spyprice['Adj_Close'][1:])/spyprice['Adj_Close'][1:]
    today_Return=(spyprice['Adj_Close'][:-1]-spyprice['Adj_Close'][1:].values)/spyprice['Adj_Close'][1:].values
    type(today_Return)

    sd_Return=today_Return.iloc[::-1].rolling(250).std().iloc[::-1]
    sd_Return=sd_Return.dropna()
    sd_Return=sd_Return[1:]
    #print(spydata)
    #print(D2spydata)
    #print(etfsmean)
    #s1=(next_Return).std()*0.1
    #s2=(next_Return).std()*2.0
    #s0=(next_Return).std()*-0.1
    #sn1=(next_Return).std()*-2.0
    classret=[ 2  if next_Return[date]>sd_Return[date]*2.0 else 1 if next_Return[date]>sd_Return[date]*0.5 else 0 if next_Return[date]>sd_Return[date]*-0.5 else -1 if next_Return[date]>sd_Return[date]*-2.0 else -2 for date in sd_Return.index]
    #classret=[ 2  if ret>s2 else 1 if ret>s1 else 0 if ret>s0 else -1 if ret>sn1 else -2 for ret in next_Return]
    classret=pd.DataFrame(classret)
    classret.index=sd_Return.index
    classret.columns=['classret']
    next_Return=pd.DataFrame(next_Return)
    today_Return=pd.DataFrame(today_Return)
    next_Return.columns=['next_Return']
    today_Return.columns=['today_Return']
    return today_Return,next_Return,classret

raw_ES_F_2015 = 'C://Users//jloss//PyCharmProjects//SMA-HullTrading-Practicum//Source Code//data//F1m2015//ES_F.txt'
raw_ES_F_2016 = 'C://Users//jloss//PyCharmProjects//SMA-HullTrading-Practicum//Source Code//data//F1m2016//ES_F.txt'
raw_ES_F_2017 = 'C://Users//jloss//PyCharmProjects//SMA-HullTrading-Practicum//Source Code//data//F1m2017//ES_F.txt'


def alldata():
    oldtime = time.time()

    colum_names = ['ticker', 'date', 'description', 'sector', 'industry', 'raw_s', 's-volume', 's-dispersion', 'raw-s-delta', 'volume-delta', 'center-date', 'center-time', 'center-time-zone']
    df_2015 = pd.read_csv(raw_ES_F_2015, skiprows = 6, sep = '\t', names = colum_names)
    df_2016 = pd.read_csv(raw_ES_F_2016, skiprows = 6, sep = '\t', names = colum_names)
    df_2017 = pd.read_csv(raw_ES_F_2017, skiprows=6, sep = '\t', names = colum_names)
    #aggregating data
    df_temp = df_2015.append(df_2016, ignore_index = True)
    df_aggregate = df_temp.append(df_2017, ignore_index = True)
    df_datetime = df_aggregate['date'].str.split(' ', n = 1, expand = True )
    df_datetime.columns = ['Date', 'Time']
    df = pd.merge(df_aggregate, df_datetime, left_index = True, right_index = True)
    #filtering based on trading hours and excluding weekends
    df = df[(df['Time'] >= '09:30:00') & (df['Time'] <= '16:00:00')]
    #excluding weekends
    #removing empty columns
    df = df.dropna(axis='columns')
    df=df.drop(columns=['ticker','date','description','center-date','center-time','center-time-zone'])
    df["volume_base_s"]=df["raw_s"]/df["s-volume"]

    #calculating ewm
    df["ewm_last20_volume_base_s"] = df.groupby("Date")["volume_base_s"].apply(lambda x: x.ewm(halflife=20).mean())
    df["ewm_last20_raw_s"] = df.groupby("Date")["raw_s"].apply(lambda x: x.ewm(halflife=20).mean())
    df["ewm_last20_s_volume"] = df.groupby("Date")["s-volume"].apply(lambda x: x.ewm(halflife=20).mean())
    df["ewm_last20_s_dispersion"] = df.groupby("Date")["s-dispersion"].apply(lambda x: x.ewm(halflife=20).mean())

    df["ewm_volume_base_s"] = df.groupby("Date")["volume_base_s"].apply(lambda x: x.ewm(halflife=390).mean())
    df["ewm_raw_s"] = df.groupby("Date")["raw_s"].apply(lambda x: x.ewm(halflife=390).mean())
    df["ewm_s_volume"] = df.groupby("Date")["s-volume"].apply(lambda x: x.ewm(halflife=390).mean())
    df["ewm_s_dispersion"] = df.groupby("Date")["s-dispersion"].apply(lambda x: x.ewm(halflife=390).mean())


    #taking the close only
    dffinal = df.groupby('Date').last().reset_index()
    dffinal.index=dffinal['Date']
    dffinal["mean_volume_base_s"] = df.groupby("Date")["volume_base_s"].sum()
    dffinal["mean_raw_s"] = df.groupby("Date")["raw_s"].sum()
    dffinal["mean_s_volume"] = df.groupby("Date")["s-volume"].sum()
    dffinal["mean_s_dispersion"] = df.groupby("Date")["s-dispersion"].sum()
    dffinal["count"] = df.groupby("Date")["raw_s"].count()
    dffinal["daily_min_raw_s"] = df.groupby("Date")["raw_s"].min()
    dffinal["daily_q1_raw_s"] = df.groupby("Date")["raw_s"].quantile(.25)
    dffinal["daily_mid_raw_s"] = df.groupby("Date")["raw_s"].quantile(.5)
    dffinal["daily_q2_raw_s"] = df.groupby("Date")["raw_s"].quantile(.75)
    dffinal["daily_max_raw_s"] = df.groupby("Date")["raw_s"].max()
    dffinal["daily_sd_raw_s"] = df.groupby("Date")["raw_s"].std()
    dffinal["daily_sd_s_volume"] = df.groupby("Date")["s-dispersion"].std()
    dffinal["daily_sd_s_dispersion"] = df.groupby("Date")["s-dispersion"].std()
    dffinal["volume_base_s_coff"] = df.groupby("Date")["volume_base_s"].apply(lambda x: (pywt.wavedec(x,'db1')[0])[0])
    dffinal["raw_s_coff"] = df.groupby("Date")["raw_s"].apply(lambda x: (pywt.wavedec(x,'db1')[0])[0])
    dffinal["s_volume_coff"] = df.groupby("Date")["s-volume"].apply(lambda x: (pywt.wavedec(x,'db1')[0])[0])
    dffinal["s_dispersion_coff"] = df.groupby("Date")["s-dispersion"].apply(lambda x: (pywt.wavedec(x,'db1')[0])[0])
    dffinal["raw_s_deg"] = df.groupby("Date")["raw_s"].apply(lambda x: len(pywt.wavedec(x,'db1')[0]))
    dffinal["s_volume_deg"] = df.groupby("Date")["s-volume"].apply(lambda x: len(pywt.wavedec(x,'db1')[0]))
    dffinal["s_dispersion_deg"] = df.groupby("Date")["s-dispersion"].apply(lambda x: len(pywt.wavedec(x,'db1')[0]))
    dffinal["raw_s_wave"] = df.groupby("Date")["raw_s"].apply(lambda x: len(pywt.wavedec(x,'db1')))
    dffinal["s_volume_wave"] = df.groupby("Date")["s-volume"].apply(lambda x: len(pywt.wavedec(x,'db1')))
    dffinal["s_dispersion_wave"] = df.groupby("Date")["s-dispersion"].apply(lambda x: len(pywt.wavedec(x,'db1')))


    today_Return,next_Return,classret=SPY()
    sd_Return=today_Return.iloc[::-1].rolling(250).std().iloc[::-1]
    sd_Return=sd_Return.dropna()
    sd_Return=sd_Return[1:]
    sd_Return
    sd_Return.columns=['sd_Return']
    dffinal = pd.concat([dffinal, next_Return,today_Return,classret,sd_Return], axis=1, sort=False,join='inner')
    dffinal['volume_base_s_delta']=(dffinal['mean_volume_base_s'][1:]-dffinal['mean_volume_base_s'][:-1].values)
    dffinal['raw-s-delta']=(dffinal['mean_raw_s'][1:]-dffinal['mean_raw_s'][:-1].values)
    dffinal['volume-delta']=(dffinal['mean_s_volume'][1:]-dffinal['mean_s_volume'][:-1].values)

    dffinal['mean_raw_s_sd']=(dffinal['mean_raw_s']-dffinal['mean_raw_s'].rolling(20).mean())/dffinal['mean_raw_s'].rolling(20).std()
    dffinal['mean_s_volume_sd']=(dffinal['mean_s_volume']-dffinal['mean_s_volume'].rolling(20).mean())/dffinal['mean_s_volume'].rolling(20).std()

    newtime = time.time()
    print('compute data time: %.3f' % (
        newtime - oldtime
    ))
    oldtime = time.time()
    return dffinal

df=alldata()
print('start stationarity')
oldtime = time.time()

testdf=df.drop(columns=['Date','raw_s','s-volume','s-dispersion','Time'])
testdf = testdf.dropna(axis='rows')


# slist=(stationarity(testdf))

newtime = time.time()
print('stationarity use: %.3fs' % (
        newtime - oldtime
    ))

