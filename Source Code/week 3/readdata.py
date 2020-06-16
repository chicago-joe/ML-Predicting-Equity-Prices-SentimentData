# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 21:35:50 2019

@author: duany
"""
import pandas as pd
import time
from yahoo_historical import Fetcher

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
        col_names=('raw_s','raw_s_mean','raw_volatility','raw_score','s','s_mean','s_volatility','s_score','s_volume','sv_mean','sv_volatility','sv_score','s_dispersion','s_buzz','s_delta','raw_s_wave','raw_s_skew','raw_s_kurt','raw_score_wave','raw_score_skew','raw_score_kurt','s_wave','s_skew','s_kurt','date')

        singleticker.columns=col_names
        for names in col_names:
            if names!= 'date':
                singleticker[names] = singleticker[names].astype(float)
        allticker[ticker]=singleticker
    return allticker


df=readdata('./meantable12ticker.txt')


tickerlist=['XLK','XLV','XLF','XLY','XLI','XLP','XLE','XLU','VNQ','GDX','SPY']

etfsmean=datatrans(tickerlist,df)

