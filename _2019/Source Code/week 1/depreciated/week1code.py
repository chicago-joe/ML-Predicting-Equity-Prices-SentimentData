# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 21:35:50 2019

@author: duany
"""
import os
import pandas as pd
import time
import numpy as np 
from yahoo_historical import Fetcher
import statsmodels.api as sm
import statsmodels.formula.api as smf               
import matplotlib.pyplot as plt
import warnings

def readfilepath(REPORT_PATH):
    filelist=[[],[],[]]
    fullpathlist=[[],[],[]]
    
    for i in range(len(REPORT_PATH)):
        for dirpath, dirnames, filenames in os.walk(REPORT_PATH[i]):
            for file in filenames:
                filelist[i].append(file)
                fullpath = os.path.join(dirpath, file)
                fullpathlist[i].append(fullpath)

    return fullpathlist,filelist


def meantable(fullpathlist):
    f1 = open('/Users/jloss/PycharmProjects/SMA-HullTrading-Practicum/Source Code/meantable.txt', 'w')
    f1.write('')
    f1.close()
    totalfilecount=0
    for i in range(len(fullpathlist)):
        totalfilecount+=len(fullpathlist[i])
    count=0
    oldtime = time.time()
    dfout = {'init':-1}
    timekey=''
    tickerkey=''
    for i in range(len(fullpathlist)):
        for j in range(len(fullpathlist[i])):
            count+=1
            print('percent: %.3f' % (
                    count/totalfilecount*100
                ))
            tempdata=pd.read_csv(fullpathlist[i][j], sep="\t",skiprows=5)
            if len(tempdata['ticker'])>0 :
                tempdata=tempdata.set_index(pd.to_datetime(tempdata['date']))
                tickerkey=tempdata['ticker'][0]
                if dfout.get(tickerkey):
                    tempticker=''
                    dfout[tickerkey]+=1
                else:
                    tempticker=tickerkey
                    dfout[tickerkey]=1

                datelist=(np.unique(np.array([pd.to_datetime(date).date() for date in tempdata.date])))
                for k in ((datelist)):
                    tempout=tempdata[str(k)].mean()
                    tempout['date']=str(k)
                    adddata('/Users/jloss/PycharmProjects/SMA-HullTrading-Practicum/Source Code/meantable2.txt',tempticker,tempout)
                    tempticker=''

    newtime = time.time()
    print('mean table use: %.3fs' % (
            newtime - oldtime
        ))
    return dfout


def alltable(fullpathlist):
    f1 = open('./alltable2.txt', 'w')
    f1.write('')
    f1.close()
    totalfilecount=0
    for i in range(len(fullpathlist)):
        totalfilecount+=len(fullpathlist[i])
    count=0
    oldtime = time.time()
    dfout = {'init':-1}
    timekey=''
    tickerkey=''
    for i in range(len(fullpathlist)):
        for j in range(len(fullpathlist[i])):
            count+=1
            print('percent: %.3f' % (
                    count/totalfilecount*100
                ))
            tempdata=pd.read_csv(fullpathlist[i][j], sep="\t",skiprows=5)
            if len(tempdata['ticker'])>0 :
                tempdata=tempdata.set_index(pd.to_datetime(tempdata['date']))
                tickerkey=tempdata['ticker'][0]
                if dfout.get(tickerkey):
                    tempticker=''
                    dfout[tickerkey]+=1
                else:
                    tempticker=tickerkey
                    dfout[tickerkey]=1

                datelist=(np.unique(np.array([pd.to_datetime(date).date() for date in tempdata.date])))
                for k in ((datelist)):
                    tempout=[str(k)]
                    tempsubdata=tempdata[str(k)]
                    tempsubsubdata=tempsubdata.mean()
                    [tempout.append(data) for data in tempsubsubdata]
                    tempsubsubdata=tempsubdata.min()
                    [tempout.append(data) for data in tempsubsubdata]
                    tempsubsubdata=tempsubdata.max()
                    [tempout.append(data) for data in tempsubsubdata]
                    tempsubsubdata=tempsubdata.median()
                    [tempout.append(data) for data in tempsubsubdata]
                    tempsubsubdata=tempsubdata.quantile(0.25)
                    [tempout.append(data) for data in tempsubsubdata]
                    tempsubsubdata=tempsubdata.quantile(0.75)
                    [tempout.append(data) for data in tempsubsubdata]
                    tempsubsubdata=tempsubdata.std()
                    [tempout.append(data) for data in tempsubsubdata]
                    tempsubsubdata=tempsubdata.skew()
                    [tempout.append(data) for data in tempsubsubdata]
                    tempsubsubdata=tempsubdata.kurt()
                    [tempout.append(data) for data in tempsubsubdata]
                    ##dwt=pywt.wavedec(item['target'], 'db1')    signal = len(dwt[0])
                    ##auto-correlation

                    adddata('./alltable2.txt',tempticker,tempout)
                    tempticker=''

    newtime = time.time()
    print('mean table use: %.3fs' % (
            newtime - oldtime
        ))
    return dfout


def savedata(dfout):
    oldtime = time.time()
    f1 = open('./meantable2.txt', 'w')
    for i in ((dfout)):

        if i != 'init':
            f1.write('\n')
        for j in ((dfout[i])):
            if j != 'init':
                f1.write('/')
            for k in range(len(dfout[i][j])):
                if k != 0:
                    f1.write(',')
                f1.write(str(dfout[i][j][k]))
    f1.close()

    newtime = time.time()
    print('save data time: %.3f' % (
        newtime - oldtime
    ))
    oldtime = time.time()

def adddata(filename,ticker,datalist):
    f1 = open(filename, 'a+')
    if ticker!='' and ticker !='init':
        f1.write('\n')
        f1.write(ticker)
        f1.write(':')

    else:
        f1.write('/')
    for k in range(len(datalist)):
        if k != 0:
            f1.write(',')
        f1.write(str(datalist[k]))
    f1.close()

def readdata(filename):
    oldtime = time.time()
    f = open(filename)
    s = f.read()
    f.close()
    ticker = s.split('\n')
    dfout = {'init':-1}
    itercars = iter(ticker)
    next(itercars)
    for tickerdata in itercars:
        namesplit = tickerdata.split(':')
        tickername=namesplit[0]
        #print(tickername)
        if dfout.get(namesplit[0]):
            tickerout=dfout[namesplit[0]]
        else:
            tickerout = {'init':-1}
        datesdata = namesplit[1].split('/')

        for datedata in datesdata:
            colsdata=datedata.split(',')
            dataout = []
            for coldata in colsdata:
                dataout.append(coldata)
            datename=dataout[len(dataout)-1]

            tickerout[datename]=dataout
        del tickerout['init']
        dfout[tickername]=tickerout
    del dfout['init']

    newtime = time.time()
    print('read data time: %.3f' % (
        newtime - oldtime
    ))
    oldtime = time.time()
    return dfout


def meanclosetable(ticker,df):
    oldtime = time.time()

    price = Fetcher(ticker, [2015,9,1], [2017,12,31]).getHistorical()
    price=price.set_index((price['Date']))
    for timedata in df[ticker]:
        if sum(price.index==timedata)==1:
            df[ticker][timedata].append(price.loc[timedata][4])
    newtime = time.time()
    print('read data time: %.3f' % (
        newtime - oldtime
    ))
    return df


def singletickeroutput(ticker):
    warnings.filterwarnings("ignore")

    singleticker=pd.DataFrame.from_dict(newdf[ticker])
    singleticker=singleticker.transpose()
    col_names=('raw_s','raw_s_mean','raw_volatility','raw_score','s','s_mean','s_volatility','s_score','s_volume','sv_mean','sv_volatility','sv_score','s_dispersion','s_buzz','s_delta','date','price')

    singleticker.columns=col_names
    for names in col_names:
        if names!= 'date':
            singleticker[names] = singleticker[names].astype(float)

    price=singleticker['price']
    returns=price.diff()
    returns[0]=0
    singleticker['returns']=returns

    est_price = smf.ols(formula='price~raw_s+raw_s_mean+raw_volatility+raw_score+s+\
                  s_mean+s_volatility+s_score+s_volume+sv_mean+sv_volatility+sv_score+s_dispersion+\
                  +s_buzz+s_delta', data=singleticker).fit()
    est_return = smf.ols(formula='returns~raw_s+raw_s_mean+raw_volatility+raw_score+s+\
                  s_mean+s_volatility+s_score+s_volume+sv_mean+sv_volatility+sv_score+s_dispersion+\
                  +s_buzz+s_delta', data=singleticker).fit()
    #the price is highly related to volatilities
    #the return is highly related to the s_delta

    print(est_price.summary())
    print(est_return.summary())
    #print(est_price.params)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_regression
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import classification_report

    forest = RandomForestClassifier(random_state=42)
    para = {'max_depth':range(1,15), 'min_samples_leaf':range(1,15)}

    CV_forest= GridSearchCV(forest,para,cv=6, n_jobs= 4,iid = True,refit= True)
    singleticker_new=singleticker
    singleticker_new=singleticker_new.drop(columns=['price','returns','date'])

    class_return=[-1]*len(returns)
    for i in range(len(returns)):
        if returns[i]>=0:
            class_return[i]=1
        else:
            class_return[i]=0

    CV_forest.fit(singleticker_new, class_return)
    best_leaf = CV_forest.best_params_['min_samples_leaf']
    best_depth = CV_forest.best_params_['max_depth']
    forest_best = RandomForestClassifier(random_state=42,min_samples_leaf=best_leaf,max_depth=best_depth,n_jobs=-1)
    forest_best.fit(singleticker_new, class_return)

    y_pred_rf = forest_best.predict(singleticker_new)

    importances = forest_best.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.title('Feature Importance')
    plt.bar(range(singleticker_new.shape[1]), importances[indices], align='center')
    plt.xticks(range(singleticker_new.shape[1]),singleticker_new.columns[indices], rotation=90)
    plt.xlim([-1, singleticker_new.shape[1]])
    plt.tight_layout()
    plt.show()

    print(classification_report(class_return, y_pred_rf))

def datainsight(tickerlist,newdf):

    for ticker in tickerlist:
        singleticker=pd.DataFrame.from_dict(newdf[ticker])
        singleticker=singleticker.transpose()
        col_names=('raw_s','raw_s_mean','raw_volatility','raw_score','s','s_mean','s_volatility','s_score','s_volume','sv_mean','sv_volatility','sv_score','s_dispersion','s_buzz','s_delta','date','price')

        singleticker.columns=col_names
        for names in col_names:
            if names!= 'date':
                singleticker[names] = singleticker[names].astype(float)

        ##XLP has a interesting graph though.
        #for col in col_names[:-2]:
        for col in ['raw_s','raw_s_mean','s_buzz']:
            fig, ax1 = plt.subplots()

            color = 'tab:red'
            ax1.set_xlabel(ticker)
            ax1.set_ylabel(col, color=color)
            ax1.plot(singleticker[col], color=color)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:blue'
            ax2.set_ylabel(ticker+' price', color=color)  # we already handled the x-label with ax1
            ax2.plot(singleticker['price'], color=color)
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            plt.xticks(np.arange(0,len(singleticker[col]),20.0))

            plt.show()


####data

df=readdata('/Users/jloss/PycharmProjects/SMA-HullTrading-Practicum/Source Code/meantable2.txt')
tickerlist=['XLK','XLV','XLF','XLY','XLI','XLP','XLE','XLU','VNQ','GDX','VOX']
for ticker in tickerlist: 
    df=meanclosetable(ticker,df)

newdf={}
for ticker in tickerlist: 
    newdf[ticker] = dict((k, v) for k, v in df[ticker].items() if len(v) >= 17)



####single ticker output
warnings.filterwarnings("ignore")
singletickeroutput('XLK')



#####plot and time-series analysis
plt.rcParams['figure.dpi'] = 120

tickerl=['XLK','XLF','XLY']
datainsight(tickerl,newdf)


        
from statsmodels.tsa.stattools import adfuller    
def adf_test(timeseries):
    print('Results of Augment Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
    
import statsmodels.api as sm
def acf_pacf_plot(ts_log_diff):
    sm.graphics.tsa.plot_acf(ts_log_diff,lags=40) #ARIMA,q
    sm.graphics.tsa.plot_pacf(ts_log_diff,lags=40) #ARIMA,p
    
adf_test(singleticker['raw_s'])
acf_pacf_plot(singleticker['raw_s'])


