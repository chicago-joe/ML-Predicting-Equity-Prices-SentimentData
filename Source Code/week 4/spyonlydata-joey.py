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
import os
import pandas as pd
import time
import numpy as np 
import pywt
from datetime import timedelta
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


#from yahoo_historical import Fetcher
'''
function
=============
find the path of all file in the folder
=============
input
=============
#REPORT_PATH: datafile forder list ex:['./F1m2015/','./F1m2016/','./F1m2017/']
#limit:       list of target ticker 
=============
output
=============
#fullpathlist: 2Dlist of full path
#filelist:     2Dlist of file name
=============
'''
def readfilepath(REPORT_PATH,limit=[]):
    filelist=[[],[],[]]
    fullpathlist=[[],[],[]]
    
    for i in range(len(REPORT_PATH)):
        for dirpath, dirnames, filenames in os.walk(REPORT_PATH[i]):
            for file in filenames:
                if len(limit)!=0:
                    ticker=file.split('.')[0]
                    if ticker not in limit:
                        #print(ticker)

                        continue 
                filelist[i].append(file)
                fullpath = os.path.join(dirpath, file)
                fullpathlist[i].append(fullpath)

    return fullpathlist,filelist


'''
function
=============
create a txt './SPYdailydata.txt' that save avg per day
=============
input
=============
#fullpathlist: 2Dlist of full path
=============
output
=============
#dfout:        list show count of each ticker
=============
'''
def SPYticker(fullpathlist):
    f1 = open('./SPYdailydata.txt', 'w')
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
                    tempticker=tickerkey
                    dfout[tickerkey]+=1
                else:
                    tempticker=tickerkey
                    dfout[tickerkey]=1

                datelist=(np.unique(np.array([pd.to_datetime(date).date() for date in tempdata.date]))) 
                for k in ((datelist)):
                    tempout=tempdata[str(k)+' 15:50:59':str(k)+' 16:00:01'].mean()
                    tempout['date']=str(k)
                    if not tempout.isnull().values.any():
                        adddata('./SPYdailydata.txt',tempticker,tempout)
                    tempticker=''
        
    newtime = time.time()
    print('mean table use: %.3fs' % (
            newtime - oldtime
        ))
    return dfout

'''
function
=============
create a txt './alltable2.txt' that save all type of numbers per day
=============
input
=============
#fullpathlist: 2Dlist of full path
=============
output
=============
#dfout:        list show count of each ticker
=============
'''
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

'''
function
=============
write data to file
=============
input
=============
#filename:     the file name
#ticker:       the ticker name
#datalist:     the data need to write
=============
output
=============
=============
'''
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
 
'''
function
=============
read data form file
=============
input
=============
#filename:     the file name
=============
output
=============
#dfout:        3D dict data [ticker][date][colnumber]
=============
'''
def readdata(filename):
    oldtime = time.time()
    f = open(filename)
    s = f.read()
    f.close()
    ticker = s.split('\n')
    dfout = {} 
    itercars = iter(ticker)
    next(itercars)
    for tickerdata in itercars:
        namesplit = tickerdata.split(':')
        tickername=namesplit[0]
        #print(tickername)
        if dfout.get(namesplit[0]):
            tickerout=dfout[namesplit[0]]
        else:
            tickerout = {}
        datesdata = namesplit[1].split('/')
        
        for datedata in datesdata:
            colsdata=datedata.split(',')
            dataout = []
            for coldata in colsdata:
                dataout.append(coldata)
            datename=dataout[len(dataout)-1]
            
            tickerout[datename]=dataout
        dfout[tickername]=tickerout

    newtime = time.time()
    print('read data time: %.3f' % (
        newtime - oldtime
    ))
    oldtime = time.time()
    return dfout

'''
function
=============
add price of the etf to the data
=============
input
=============
#ticker:       the ticker name
#df:           the data need edit
=============
output
=============
#df:           the data after edit
=============
'''
# def meanclosetable(ticker,df):
#     oldtime = time.time()
# 
#     price = Fetcher(ticker, [2015,9,1], [2017,12,31]).getHistorical()
#     price=price.set_index((price['Date']))
#     for timedata in df[ticker]:
#         if sum(price.index==timedata)==1:
#             df[ticker][timedata].append(price.loc[timedata][4])
#     newtime = time.time()
#     print('update data time: %.3f' % (
#         newtime - oldtime
#     ))
#     return df
# 
'''
function
=============
select the date that has both factor and price
=============
input
=============
#tickerlist:   the ticker name list
#df:           the data need edit
=============
output
=============
#newdf:        the data after edit
=============
'''
def submeanclosetableonly(tickerlist,df):
    newdf={}
    for ticker in tickerlist: 
        newdf[ticker] = dict((k, v) for k, v in df[ticker].items() if len(v) >= 17)
    return newdf


'''
function
=============
make data easier to use
=============
input
=============
#tickerlist:   the ticker name list
#df:           the data need edit
=============
output
=============
#allticker:    the data after edit
=============
'''
def datatrans(tickerlist,df):
    #for ticker in tickerlist: 
    #    df=meanclosetable(ticker,df)
    #newdf=submeanclosetableonly(tickerlist,df)
    allticker={}
    
    for ticker in tickerlist: 
        singleticker=df[ticker]
        if singleticker.get('nan'):
            del singleticker['nan']
        singleticker=pd.DataFrame.from_dict(df[ticker])
        singleticker=singleticker.transpose()
        col_names=('raw_s','raw_s_mean','raw_volatility','raw_score','s','s_mean','s_volatility','s_score','s_volume','sv_mean','sv_volatility','sv_score','s_dispersion','s_buzz','s_delta','date')
        
        singleticker.columns=col_names
        for names in col_names:
            if names!= 'date':
                singleticker[names] = singleticker[names].astype(float)
        singleticker['s_delta']=(singleticker['raw_s'][1:]-singleticker['raw_s'][:-1].values)

        allticker[ticker]=singleticker
    return allticker
       

def SPY():
    
    
    spyprice=pd.read_csv('SPY Price Data.csv')
    spyprice.index=spyprice['Date']
    next_Return=(spyprice['Adj_Close'][:-1].values-spyprice['Adj_Close'][1:])/spyprice['Adj_Close'][1:]
    today_Return=(spyprice['Adj_Close'][:-1]-spyprice['Adj_Close'][1:].values)/spyprice['Adj_Close'][1:].values
    
    
    #print(spydata)
    #print(D2spydata)
    #print(etfsmean)
    s1=(next_Return).std()*0.1
    s2=(next_Return).std()*2.0
    s0=(next_Return).std()*-0.1
    sn1=(next_Return).std()*-2.0
    classret=[ 2  if ret>s2 else 1 if ret>s1 else 0 if ret>s0 else -1 if ret>sn1 else -2 for ret in next_Return]
    classret=pd.DataFrame(classret)
    classret.index=next_Return.index
    classret.columns=['classret']
    next_Return=pd.DataFrame(next_Return)
    today_Return=pd.DataFrame(today_Return)
    next_Return.columns=['next_Return']
    today_Return.columns=['today_Return']
    return today_Return,next_Return,classret

def mix():
    result=pd.DataFrame()
    for ticker in etfsmean:
        if len(result)==0:
            oneticker=etfsmean[ticker]
    
            col_names=['raw_s','raw_s_mean','raw_volatility','raw_score','s','s_mean','s_volatility','s_score','s_volume','sv_mean','sv_volatility','sv_score','s_dispersion','s_buzz','s_delta']
            oneticker = oneticker.drop(columns='date')
            oneticker.columns=[names+ticker for names in col_names]
            result=oneticker
            continue
        oneticker=etfsmean[ticker]
        col_names=['raw_s','raw_s_mean','raw_volatility','raw_score','s','s_mean','s_volatility','s_score','s_volume','sv_mean','sv_volatility','sv_score','s_dispersion','s_buzz','s_delta']
        oneticker = oneticker.drop(columns='date')
        oneticker.columns=[names+ticker for names in col_names]
        result = pd.concat([result,oneticker], axis=1, sort=False,join='inner')
    result = pd.concat([result, next_Return,today_Return,classret], axis=1, sort=False,join='inner')
    return result


def write_svolume_dailyclose_to_csv():
    SPY_s_Volume = pd.DataFrame(SPYdailydata['s_volumeSPY'])

    # write all days and work in excel
    all=pd.DataFrame.from_dict(df)
    all.head()
    all.to_csv('all.csv')
    # write weekdays (no weekends) and work in excel
    bdays = SPYdailydata
    bdays.head()
    bdays.to_csv('bdays.csv')

    # store attributes of s-Volume in dataframes for future research and analysis:
    SPY_sv_Mean = pd.DataFrame(SPYdailydata['sv_meanSPY'])
    SPY_sv_Volatility = pd.DataFrame(SPYdailydata['sv_volatilitySPY'])
    SPY_sv_Score = pd.DataFrame(SPYdailydata['sv_scoreSPY'])
    return

def analyze_svolume_dailyclose_data():
    # make a new dataframe to work in from df
    tmp = pd.DataFrame.from_dict(df.get('SPY').values())

    # now we are going to add the column names
    tmpwip = tmp
    col_names=['raw_s','raw_s_mean','raw_volatility','raw_score','s','s_mean','s_volatility',
               's_score','s_volume','sv_mean','sv_volatility','sv_score','s_dispersion','s_buzz','s_delta']
    col_names.append('date')
    print(col_names)

    # this also changes the column names in tmp:
    tmpwip.columns = col_names

#     now we need to change the index to dates instead of numbers:
    tmp = tmp.set_index('date')
    tmp.head()

#     next, we need to convert the dataframe's dtype from non-null objects to numeric values
#     otherwise we can't do anything with the data:
    tmp.info()
    myNewDataFrame = tmp.apply(pd.to_numeric)

    # now this should say non-null float64, we can work on the dataframe now!
    myNewDataFrame.info()

#    plot s_volume data
    myNewDataFrame[['s_volume']].plot(figsize=(20,10),linewidth=1,fontsize=16)
    plt.show()

    # let's just make a series for s_volume and nothing else
    myNewSvolume = myNewDataFrame[['s_volume']]

#    Let's do First-Order Differencing to investigate Seasonal Trends in the data
#    Link: https://www.datacamp.com/community/tutorials/time-series-analysis-tutorial
    myNewSvolume.diff().plot()



tickerlist=['ES_F']
fullpathlist,filelist=readfilepath(['./F1m2015/','./F1m2016/','./F1m2017/'],tickerlist)

#dfout=SPYticker(fullpathlist)
df=readdata('./SPYdailydata.txt')

etfsmean=datatrans(tickerlist,df)

today_Return,next_Return,classret=SPY()

SPYdailydata=mix()
SPYdailydata['s_deltaSPY']=(SPYdailydata['raw_sSPY'][1:]-SPYdailydata['raw_sSPY'][:-1].values)


