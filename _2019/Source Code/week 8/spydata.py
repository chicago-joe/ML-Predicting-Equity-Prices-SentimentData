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
from scipy.stats.mstats import winsorize

import seaborn as sns; sns.set()
import pandas as pd
import time
import numpy as np 
import pywt #导入PyWavelets
from datetime import timedelta  
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller   

def oneheat(result):
    plt.subplots(figsize=(15,10))
    corr = pd.DataFrame(np.corrcoef(result.transpose()))
    corr.index=result.columns
    corr.columns=result.columns
    
    fig=sns.heatmap(corr,xticklabels=False, yticklabels=1, vmin=-1, vmax=1,center=0).get_figure()    
    fig.dpi=200
    fig.savefig('./heatmap/spyfator.png')

def acf_pacf_plot(ts_log_diff,name):
    sm.graphics.tsa.plot_acf(ts_log_diff,lags=40) #ARIMA,q
    #plt.show()    
    plt.savefig('./station/'+name+'acf.png')
#    sm.graphics.tsa.plot_pacf(ts_log_diff,lags=40) #ARIMA,p
#    #plt.show()    
#    plt.savefig('./station/'+name+'pacf.png')
    fig, ax1 = plt.subplots()
    
    ax1.set_title(name)
    ax1.set_xlabel('time (D)')
    ax1.set_ylabel(name)
    ax1.plot(ts_log_diff)
    ax1.tick_params(axis='y')
    #ax1.set_ylim([-abs(max(df[linex], key=abs))*1.1, abs(max(df[linex], key=abs))*1.1])
    
    plt.xticks(np.arange(0,len(df.index),len(df.index)/5))
    plt.title=name
    ymin=ts_log_diff.min()-0.1*ts_log_diff.std()
    ymax=ts_log_diff.max()+0.1*ts_log_diff.std()
    ax1.set_ylim([ymin,ymax])
    #fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig('./factor/'+name+'.png')
    plt.show()
    
    
    fig, ax1 = plt.subplots()
    
    ax1.set_title(name)
    ax1.set_xlabel('time (D)')
    ax1.set_ylabel(name)
    ts_log_diff=ts_log_diff[-100:]
    ax1.plot(ts_log_diff)
    ax1.tick_params(axis='y')
    #ax1.set_ylim([-abs(max(df[linex], key=abs))*1.1, abs(max(df[linex], key=abs))*1.1])
    
    plt.xticks(np.arange(0,len(ts_log_diff.index),len(ts_log_diff.index)/5))
    plt.title=name
    ax1.set_ylim([ymin,ymax])

    #fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig('./factorlast5m/'+name+'.png')
    plt.show()
def stationarity(result):
    plist={}
    for col in result:
        acf_pacf_plot(result[col],col)
        if adf_test(result[col])['p-value']<0.05:
            st=True
        else:
            st=False
        plist[col]=st
    return plist

def adf_test(timeseries):
    #print('Results of Augment Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return(dfoutput)

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
    
    todayclassret=[ 2  if today_Return[date]>sd_Return[date]*2.0 else 1 if today_Return[date]>sd_Return[date]*0.5 else 0 if today_Return[date]>sd_Return[date]*-0.5 else -1 if today_Return[date]>sd_Return[date]*-2.0 else -2 for date in sd_Return.index]
    #classret=[ 2  if ret>s2 else 1 if ret>s1 else 0 if ret>s0 else -1 if ret>sn1 else -2 for ret in next_Return]
    todayclassret=pd.DataFrame(todayclassret)
    todayclassret.index=sd_Return.index
    todayclassret.columns=['todayclassret']
    
    next_Return=pd.DataFrame(next_Return)
    today_Return=pd.DataFrame(today_Return)
    next_Return.columns=['next_Return']
    today_Return.columns=['today_Return']
    return today_Return,next_Return,classret,todayclassret


def alldata():
    oldtime = time.time()
    
    colum_names = ['ticker', 'date', 'description', 'sector', 'industry', 'raw_s', 's-volume', 's-dispersion', 'raw-s-delta', 'volume-delta', 'center-date', 'center-time', 'center-time-zone']
    df_2015 = pd.read_csv('SPY2015ActFeed.txt', skiprows = 6, sep = '\t', names = colum_names)
    df_2016 = pd.read_csv('SPY2016ActFeed.txt', skiprows = 6, sep = '\t', names = colum_names)
    df_2017 = pd.read_csv('SPY2017ActFeed.txt', skiprows=6, sep = '\t', names = colum_names)
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
    df["ewm_last20_volume_base_s"] = df.groupby("Date")["volume_base_s"].apply(lambda x: x.ewm(span=20).mean())
    df["ewm_last20_raw_s"] = df.groupby("Date")["raw_s"].apply(lambda x: x.ewm(span=20).mean())
    df["ewm_last20_s_volume"] = df.groupby("Date")["s-volume"].apply(lambda x: x.ewm(span=20).mean())
    df["ewm_last20_s_dispersion"] = df.groupby("Date")["s-dispersion"].apply(lambda x: x.ewm(span=20).mean())
    
    df["ewm_volume_base_s"] = df.groupby("Date")["volume_base_s"].apply(lambda x: x.ewm(span=390).mean())
    df["ewm_raw_s"] = df.groupby("Date")["raw_s"].apply(lambda x: x.ewm(span=390).mean())
    df["ewm_s_volume"] = df.groupby("Date")["s-volume"].apply(lambda x: x.ewm(span=390).mean())
    df["ewm_s_dispersion"] = df.groupby("Date")["s-dispersion"].apply(lambda x: x.ewm(span=390).mean())
    
    
    #taking the close only
    dffinal = df.groupby('Date').last().reset_index()
    dffinal.index=dffinal['Date']
    dffinal["mean_volume_base_s"] = df.groupby("Date")["volume_base_s"].mean()
    dffinal["mean_raw_s"] = df.groupby("Date")["raw_s"].mean()
    dffinal["mean_s_volume"] = df.groupby("Date")["s-volume"].mean()
    dffinal["mean_s_dispersion"] = df.groupby("Date")["s-dispersion"].mean()
    dffinal["count"] = df.groupby("Date")["raw_s"].count()
    dffinal["daily_min_raw_s"] = df.groupby("Date")["raw_s"].min()
    dffinal["daily_q1_raw_s"] = df.groupby("Date")["raw_s"].quantile(.25)
    dffinal["daily_mid_raw_s"] = df.groupby("Date")["raw_s"].quantile(.5)
    dffinal["daily_q2_raw_s"] = df.groupby("Date")["raw_s"].quantile(.75)
    dffinal["daily_max_raw_s"] = df.groupby("Date")["raw_s"].max()
    dffinal["daily_sd_volume_base_s"] = df.groupby("Date")["volume_base_s"].std()
    dffinal["daily_sd_raw_s"] = df.groupby("Date")["raw_s"].std()
    dffinal["daily_sd_s_volume"] = df.groupby("Date")["s-volume"].std()
    dffinal["daily_sd_s_dispersion"] = df.groupby("Date")["s-dispersion"].std()
    dffinal["volume_base_s_coff"] = df.groupby("Date")["volume_base_s"].apply(lambda x: (pywt.wavedec(x,'db1')[0])[0])
    dffinal["raw_s_coff"] = df.groupby("Date")["raw_s"].apply(lambda x: (pywt.wavedec(x,'db1')[0])[0])
    dffinal["s_volume_coff"] = df.groupby("Date")["s-volume"].apply(lambda x: (pywt.wavedec(x,'db1')[0])[0])
    dffinal["s_dispersion_coff"] = df.groupby("Date")["s-dispersion"].apply(lambda x: (pywt.wavedec(x,'db1')[0])[0])
    dffinal["volume_base_s_deg"] = df.groupby("Date")["volume_base_s"].apply(lambda x: len(pywt.wavedec(x,'db1')[0]))
    dffinal["raw_s_deg"] = df.groupby("Date")["raw_s"].apply(lambda x: len(pywt.wavedec(x,'db1')[0]))
    dffinal["s_volume_deg"] = df.groupby("Date")["s-volume"].apply(lambda x: len(pywt.wavedec(x,'db1')[0]))
    dffinal["s_dispersion_deg"] = df.groupby("Date")["s-dispersion"].apply(lambda x: len(pywt.wavedec(x,'db1')[0]))
    dffinal["volume_base_s_wave"] = df.groupby("Date")["volume_base_s"].apply(lambda x: len(pywt.wavedec(x,'db1')))
    dffinal["raw_s_wave"] = df.groupby("Date")["raw_s"].apply(lambda x: len(pywt.wavedec(x,'db1')))
    dffinal["s_volume_wave"] = df.groupby("Date")["s-volume"].apply(lambda x: len(pywt.wavedec(x,'db1')))
    dffinal["s_dispersion_wave"] = df.groupby("Date")["s-dispersion"].apply(lambda x: len(pywt.wavedec(x,'db1')))
    
    
    today_Return,next_Return,classret,todayclassret=SPY()
    sd_Return=today_Return.iloc[::-1].rolling(250).std().iloc[::-1]
    sd_Return=sd_Return.dropna()
    sd_Return=sd_Return[1:]
    sd_Return
    sd_Return.columns=['sd_Return']
    dffinal = pd.concat([dffinal, next_Return,today_Return,classret,sd_Return,todayclassret], axis=1, sort=False,join='inner')
    dffinal['volume_base_s_delta']=(dffinal['mean_volume_base_s'][1:]-dffinal['mean_volume_base_s'][:-1].values)
    dffinal['raw-s-delta']=(dffinal['mean_raw_s'][1:]-dffinal['mean_raw_s'][:-1].values)
    dffinal['volume-delta']=(dffinal['mean_s_volume'][1:]-dffinal['mean_s_volume'][:-1].values)
    dffinal['s_dispersion_delta']=(dffinal['mean_s_dispersion'][1:]-dffinal['mean_s_dispersion'][:-1].values)
    
    dffinal['volume_base_s_z']=(dffinal['mean_volume_base_s']-dffinal['mean_volume_base_s'].rolling(26).mean())/dffinal['mean_volume_base_s'].rolling(26).std()
    dffinal['raw_s_z']=(dffinal['mean_raw_s']-dffinal['mean_raw_s'].rolling(26).mean())/dffinal['mean_raw_s'].rolling(26).std()
    dffinal['s_volume_z']=(dffinal['mean_s_volume']-dffinal['mean_s_volume'].rolling(26).mean())/dffinal['mean_s_volume'].rolling(26).std()
    dffinal['s_dispersion_z']=(dffinal['mean_s_dispersion']-dffinal['mean_s_dispersion'].rolling(26).mean())/dffinal['mean_s_dispersion'].rolling(26).std()
    
    dffinal['volume_base_s_MACD_ewma12-ewma26'] = dffinal["mean_volume_base_s"].ewm(span=12).mean() - dffinal["mean_volume_base_s"].ewm(span=26).mean()
    dffinal['raw_s_MACD_ewma12-ewma26'] = dffinal["mean_raw_s"].ewm(span=12).mean() - dffinal["mean_raw_s"].ewm(span=26).mean()
    dffinal['s_volume_MACD_ewma12-ewma26'] = dffinal["mean_s_volume"].ewm(span=12).mean() - dffinal["mean_s_volume"].ewm(span=26).mean()
    dffinal['s_dispersion_MACD_ewma12-ewma26'] = dffinal["mean_s_dispersion"].ewm(span=12).mean() - dffinal["mean_s_dispersion"].ewm(span=26).mean()
    
    newtime = time.time()
    print('compute data time: %.3f' % (
        newtime - oldtime
    ))
    oldtime = time.time()

    return dffinal
def using_mstats(s):
    return winsorize(s, limits=[0.005, 0.005])

df=alldata()

print('start stationarity')
oldtime = time.time()

testdf=df.drop(columns=['Date','raw_s','s-volume','s-dispersion','Time','volume_base_s'])
testdf = testdf.dropna(axis='rows')
testdfold=testdf

#testdf=testdf.apply(using_mstats, axis=0)
#testdfstd=testdf.rolling(26).std()
#testdfmean=testdf.rolling(26).mean()
##testdfstd=testdfstd.dropna()
##testdfmean=testdfmean.dropna()
#testdfmin=testdfmean-3*testdfstd
#testdfmax=testdfmean+3*testdfstd
#
#
#table=(testdfmin>testdf)
#table.to_csv('table.csv', sep=',')
#
#table2=(testdfmax<testdf)
#table2.to_csv('table2.csv', sep=',')
#
#table1=table | table2
#table1.to_csv('table1.csv', sep=',')

#testdf.sum()
#testdfmin[table].sum()
#testdf[table]=testdfmin[table]
#testdf[table2]=testdfmax[table2]
df_train=testdf[0:464]
df_test=testdf[464:]
train_y=df_train['next_Return']
test_y=df_test['next_Return']

df_train=df_train.apply(using_mstats, axis=0)

slist=(stationarity(df_train))

newtime = time.time()
print('stationarity use: %.3fs' % (
        newtime - oldtime
    ))


print('start outlier')
oldtime = time.time()
slist=pd.DataFrame(slist, index=[0])
factors=[]
for i in slist.columns:
    if slist[i][0]==1:
        factors.append(i)
#factors.remove("volume_base_s_deg")
factors.remove("raw_s_deg")
factors.remove("s_volume_deg")
factors.remove("s_dispersion_deg")
factors.remove("classret")
factors.remove("next_Return")

train_x=df_train[factors]


maxtrain=df_train.max()
mintrain=df_train.min()
for col in df_test:
    df_test[col][df_test[col] < mintrain[col]] = mintrain[col]
    df_test[col][df_test[col] > maxtrain[col]] = maxtrain[col]
    
tablemin=(mintrain>df_test)
tablemin.to_csv('tablemin.csv', sep=',')

tablemax=(maxtrain<df_test)
tablemax.to_csv('tablemax.csv', sep=',')

table1=tablemax | tablemin
table1.to_csv('table1.csv', sep=',')

test_x=df_test[factors]

newtime = time.time()
print('outlier use: %.3fs' % (
        newtime - oldtime
    ))

print('start heatmap')
oldtime = time.time()
oneheat(train_x)
newtime = time.time()
print('heatmap use: %.3fs' % (
        newtime - oldtime
    ))


train_x.to_csv('train_x.txt', sep=',')
train_y.to_csv('train_y.txt', sep=',')
test_x.to_csv('test_x.txt', sep=',')
test_y.to_csv('test_y.txt', sep=',')

















