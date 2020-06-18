# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 21:35:50 2019

@author: duany
"""
import os
import pandas as pd
import time
import numpy as np 
#from yahoo_historical import Fetcher

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


def meantable(fullpathlist):
    f1 = open('./meantable2.txt', 'w')
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
                    tempout=tempdata[str(k)].mean()
                    tempout['date']=str(k)
                    adddata('./meantable2.txt',tempticker,tempout)
                    tempticker=''
        
    newtime = time.time()
    print('mean table use: %.3fs' % (
            newtime - oldtime
        ))
    return dfout


def meantable12ticker(fullpathlist):
    f1 = open('./meantable12ticker.txt', 'w')
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
                    tempout=tempdata[str(k)].mean()
                    tempout['date']=str(k)
                    adddata('./meantable12ticker.txt',tempticker,tempout)
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


def meanclosetable(ticker,df):
    oldtime = time.time()

    price = Fetcher(ticker, [2015,9,1], [2017,12,31]).getHistorical()
    price=price.set_index((price['Date']))
    for timedata in df[ticker]:
        if sum(price.index==timedata)==1:
            df[ticker][timedata].append(price.loc[timedata][4])
    newtime = time.time()
    print('update data time: %.3f' % (
        newtime - oldtime
    ))
    return df

def submeanclosetableonly(tickerlist,df):
    newdf={}
    for ticker in tickerlist: 
        newdf[ticker] = dict((k, v) for k, v in df[ticker].items() if len(v) >= 17)
    return newdf

fullpathlist,filelist=readfilepath(['./data/2015/','./data/2016/','./data/2017/'])

#dfout=meantable(fullpathlist)

       
#savedata(dfout)

df=readdata('./meantable2.txt')

tickerlist=['XLK','XLV','XLF','XLY','XLI','XLP','XLE','XLU','VNQ','GDX','VOX','SPY']

<<<<<<< HEAD:Source Code/matrix-v2.py
fullpathlist,filelist=readfilepath(['./data/2015/','./data/2016/','./data/2017/'],tickerlist)
# dfout=meantable12ticker(fullpathlist)
=======
fullpathlist,filelist=readfilepath(['./2015/','./2016/','./2017/'],tickerlist)
#dfout=meantable12ticker(fullpathlist)
>>>>>>> Rain:Source Code/week 1/matrix-v2.py

df=readdata('./meantable12ticker.txt')

#for ticker in tickerlist: 
#    df=meanclosetable(ticker,df)

#newdf=submeanclosetableonly(tickerlist,df)

#newdf2=pd.DataFrame.from_dict(newdf['XLK'])

#newdf2=newdf2.transpose()
#for ticker in df:
#    df=meanclosetable(ticker,df)



