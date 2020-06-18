# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 21:35:50 2019

@author: duany
"""
import os
import pandas as pd
import time
import numpy as np 
import pywt #导入PyWavelets
from datetime import timedelta  

#from yahoo_historical import Fetcher
'''
function
=============
find the path of all file in the folder
=============
input
=============
#REPORT_PATH: datafile forder list ex:['./2015/','./2016/','./2017/']
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
create a txt './meantable2.txt' that save avg per day
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
                    tempout=tempdata[str(k)+' 09:29:59':str(k)+' 16:00:01'].mean()
                    tempdaydata=tempdata[str(k)+' 09:29:59':str(k)+' 16:00:01']['raw-s']
                    tempwave=pywt.wavedec(tempdaydata, 'db1')
                    if len(tempwave)>0:
                        if len(tempwave[0])>0:
                            tempout['raw_s_wave']= (tempwave[0][0])
                        else:
                            tempout['raw_s_wave']= 0
                    else:
                        tempout['raw_s_wave']= 0

                    tempout['raw_s_skew']=(tempdaydata).skew()
                    tempout['raw_s_kurt']=(tempdaydata).kurt()-3.0
                    tempdaydata=tempdata[str(k)+' 09:29:59':str(k)+' 16:00:01']['raw-score']
                    tempwave=pywt.wavedec(tempdaydata, 'db1')
                    if len(tempwave)>0:
                        if len(tempwave[0])>0:
                            tempout['raw_score_wave']= (tempwave[0][0])
                        else:
                            tempout['raw_score_wave']= 0
                    else:
                        tempout['raw_score_wave']= 0
                    tempout['raw_score_skew']=(tempdaydata).skew()
                    tempout['raw_score_kurt']=(tempdaydata).kurt()-3.0
                    tempdaydata=tempdata[str(k- timedelta(days=30))+' 09:29:59':str(k)+' 16:00:01']['s']
                    tempwave=pywt.wavedec(tempdaydata, 'db1')
                    if len(tempwave)>0:
                        if len(tempwave[0])>0:
                            tempout['s_wave']= (tempwave[0][0])
                        else:
                            tempout['s_wave']= 0
                    else:
                        tempout['s_wave']= 0
                    tempout['s_skew']=(tempdaydata).skew()
                    tempout['s_kurt']=(tempdaydata).kurt()-3.0
                    tempout['date']=str(k)
                    if not tempout.isnull().values.any():
                        adddata('./meantable2.txt',tempticker,tempout)
                    tempticker=''
        
    newtime = time.time()
    print('mean table use: %.3fs' % (
            newtime - oldtime
        ))
    return dfout

'''
function
=============
create a txt './meantable12ticker.txt' that save avg per day
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
                    tempout=tempdata[str(k)+' 09:29:59':str(k)+' 16:00:01'].mean()
                    tempdaydata=tempdata[str(k)+' 09:29:59':str(k)+' 16:00:01']['raw-s']
                    tempwave=pywt.wavedec(tempdaydata, 'db1')
                    if len(tempwave)>0:
                        if len(tempwave[0])>0:
                            tempout['raw_s_wave']= (tempwave[0][0])
                        else:
                            tempout['raw_s_wave']= 0
                    else:
                        tempout['raw_s_wave']= 0

                    tempout['raw_s_skew']=(tempdaydata).skew()
                    tempout['raw_s_kurt']=(tempdaydata).kurt()-3.0
                    tempdaydata=tempdata[str(k)+' 09:29:59':str(k)+' 16:00:01']['raw-score']
                    tempwave=pywt.wavedec(tempdaydata, 'db1')
                    if len(tempwave)>0:
                        if len(tempwave[0])>0:
                            tempout['raw_score_wave']= (tempwave[0][0])
                        else:
                            tempout['raw_score_wave']= 0
                    else:
                        tempout['raw_score_wave']= 0
                    tempout['raw_score_skew']=(tempdaydata).skew()
                    tempout['raw_score_kurt']=(tempdaydata).kurt()-3.0
                    tempdaydata=tempdata[str(k- timedelta(days=30))+' 09:29:59':str(k)+' 16:00:01']['s']
                    tempwave=pywt.wavedec(tempdaydata, 'db1')
                    if len(tempwave)>0:
                        if len(tempwave[0])>0:
                            tempout['s_wave']= (tempwave[0][0])
                        else:
                            tempout['s_wave']= 0
                    else:
                        tempout['s_wave']= 0
                    tempout['s_skew']=(tempdaydata).skew()
                    tempout['s_kurt']=(tempdaydata).kurt()-3.0
                    tempout['date']=str(k)
                    if not tempout.isnull().values.any():
                        adddata('./meantable12ticker.txt',tempticker,tempout)
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

fullpathlist,filelist=readfilepath(['./2015/','./2016/','./2017/'])

#dfout=meantable(fullpathlist)
df=readdata('./meantable2.txt')

tickerlist=['XLK','XLV','XLF','XLY','XLI','XLP','XLE','XLU','VNQ','GDX','SPY']
fullpathlist,filelist=readfilepath(['./2015/','./2016/','./2017/'],tickerlist)
#dfout=meantable12ticker(fullpathlist)
df=readdata('./meantable12ticker.txt')

#for ticker in tickerlist: 
#    df=meanclosetable(ticker,df)

#newdf=submeanclosetableonly(tickerlist,df)

#newdf2=pd.DataFrame.from_dict(newdf['XLK'])

#newdf2=newdf2.transpose()
#for ticker in df:
#    df=meanclosetable(ticker,df)



