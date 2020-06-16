# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:28:21 2019

@author: duany
"""
import readdata
import spy15mindata
import seaborn as sns; sns.set()
import pandas as pd
import numpy as np 


def SPY():
    
    fullpathlist,filelist=spy15mindata.readfilepath(['./2015/','./2016/','./2017/'],['SPY'])
    D2spydata=spy15mindata.SPYdata(fullpathlist)
    spydata=spy15mindata.D2_D1(D2spydata)
    
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

def heatmap():
    
    for ticker in etfsmean:
        oneticker=etfsmean[ticker]
        result = pd.concat([oneticker, next_Return,today_Return,classret], axis=1, sort=False,join='inner')
        result = result.drop(columns='date')
        #ax=sns.heatmap(result, center=0)
        corr = pd.DataFrame(np.corrcoef(result.transpose()))
        corr.index=result.columns
        corr.columns=result.columns
        
        fig=sns.heatmap(corr,cbar=False,xticklabels=False, yticklabels=1, vmin=-1, vmax=1,center=0).get_figure()    
        fig.dpi=500
        fig.savefig('./heatmap/'+ticker+'heatmap.png')


def mix():
    result=pd.DataFrame()
    for ticker in etfsmean:
        if len(result)==0:
            oneticker=etfsmean[ticker]
    
            col_names=['raw_s','raw_s_mean','raw_volatility','raw_score','s','s_mean','s_volatility','s_score','s_volume','sv_mean','sv_volatility','sv_score','s_dispersion','s_buzz','s_delta','raw_s_wave','raw_s_skew','raw_s_kurt','raw_score_wave','raw_score_skew','raw_score_kurt','s_wave','s_skew','s_kurt']
            oneticker = oneticker.drop(columns='date')
            oneticker.columns=[names+ticker for names in col_names]
            result=oneticker
            continue
        oneticker=etfsmean[ticker]
        col_names=['raw_s','raw_s_mean','raw_volatility','raw_score','s','s_mean','s_volatility','s_score','s_volume','sv_mean','sv_volatility','sv_score','s_dispersion','s_buzz','s_delta','raw_s_wave','raw_s_skew','raw_s_kurt','raw_score_wave','raw_score_skew','raw_score_kurt','s_wave','s_skew','s_kurt']
        oneticker = oneticker.drop(columns='date')
        oneticker.columns=[names+ticker for names in col_names]
        result = pd.concat([result,oneticker], axis=1, sort=False,join='inner')
    result = pd.concat([result, next_Return,today_Return,classret], axis=1, sort=False,join='inner')
    return result

def oneheat(result):
    corr = pd.DataFrame(np.corrcoef(result.transpose()))
    corr.index=result.columns
    corr.columns=result.columns
    
    fig=sns.heatmap(corr,xticklabels=False, yticklabels=1, vmin=-1, vmax=1,center=0).get_figure()    
    fig.dpi=2000
    fig.savefig('./heatmap/allheatmap.png')

df=readdata.readdata('./meantable12ticker.txt')
tickerlist=['XLK','XLV','XLF','XLY','XLI','XLP','XLE','XLU','VNQ','GDX','SPY']
etfsmean=readdata.datatrans(tickerlist,df)

today_Return,next_Return,classret=SPY()

#heatmap()
result=mix()
#oneheat(result)

