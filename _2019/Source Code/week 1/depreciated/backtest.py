# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 21:46:38 2019

@author: duany
"""
import csv
import pandas as pd
import math
    
    
def readspclose(filename):
    with open(filename) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        dates = []
        closes = []
        next(readCSV)
        for row in readCSV:
            close = row[4]
            date = row[0]
    
            dates.append(date)
            closes.append(close)
    
        output = pd.DataFrame(closes)
        output.index=dates
        output.columns=["close"]
    return output

def backtest(data,decision,initfund,amountordir,fixfee,valuefee):
    PL=0
    amount=0
    cash = initfund
    for tran in decision:
        if amountordir=="amount":
            if initfund<0:
                PL-=tran[1]*data.loc[tran[0],'close']
                amount+=tran[1]
            else:
                if cash>=abs(tran[1]*data.loc[tran[0],'close']):
                    PL-=tran[1]*data.loc[tran[0],'close']
                    amount+=tran[1]
                    cash-=tran[1]*data.loc[tran[0],'close']
                else:
                    cash-=math.floor(cash/data.loc[tran[0],'close'])*data.loc[tran[0],'close']
                    PL-=math.floor(cash/data.loc[tran[0],'close'])*data.loc[tran[0],'close']
                    amount+=math.floor(cash/data.loc[tran[0],'close'])
        else:
            if tran[1]>0:
                cash
            elif tran[1]<0:
                if initfund<0:
                    PL-=-1*data.loc[tran[0],'close']
                    amount+=tran[1]
                else:
                    if cash>=abs(tran[1]*data.loc[tran[0],'close']):
                        PL-=tran[1]*data.loc[tran[0],'close']
                        amount+=tran[1]
                        cash-=tran[1]*data.loc[tran[0],'close']
                    else:
                        cash-=math.floor(cash/data.loc[tran[0],'close'])*data.loc[tran[0],'close']
                        PL-=math.floor(cash/data.loc[tran[0],'close'])*data.loc[tran[0],'close']
                        amount+=math.floor(cash/data.loc[tran[0],'close'])
            else:
                if initfund<0:
                    PL+=amount*data.loc[tran[0],'close']
                    amount=0
                else:
                    cash+=amount*data.loc[tran[0],'close']
                    PL+=amount*data.loc[tran[0],'close']
                    amount=0
    PL+=amount*decision[len(decision)-1][0]
    return PL

spdata=readspclose('^GSPC.csv')

#print(spdata.loc['2015-09-02','close'])

PL,message=backtest(spdata,[],100000,'dir',0,0)

