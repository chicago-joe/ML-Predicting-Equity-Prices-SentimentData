################################################################
# ensembling.py
# Ensemble Methods
# Created by Joseph Loss on 11/06/2019
#
# Contact: loss2@illinois.edu
###############################################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score,roc_auc_score,accuracy_score,explained_variance_score
#import pylab as plot
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV 
from scipy.stats.mstats import winsorize
from statsmodels.tsa.stattools import adfuller   
import warnings
import copy

warnings.simplefilter("ignore")
def SPY():
    
    
    spyprice=pd.read_csv('SPY Price Data.csv')
    spyprice.index=spyprice['Date']
    next_Return=(spyprice['Adj_Close'][:-1].values-spyprice['Adj_Close'][1:])/spyprice['Adj_Close'][1:]
    today_Return=(spyprice['Adj_Close'][:-1]-spyprice['Adj_Close'][1:].values)/spyprice['Adj_Close'][1:].values
    type(today_Return)
    
    sd_Return=today_Return.iloc[::-1].rolling(250).std().iloc[::-1]
    sd_Return=sd_Return.dropna()
    sd_Return=sd_Return[1:]

    classret=[ 2  if next_Return[date]>sd_Return[date]*1.0 else 1 if next_Return[date]>sd_Return[date]*0.05 else 0 if next_Return[date]>sd_Return[date]*-0.05 else -1 if next_Return[date]>sd_Return[date]*-1.0 else -2 for date in sd_Return.index]
    #classret=[ 2  if ret>s2 else 1 if ret>s1 else 0 if ret>s0 else -1 if ret>sn1 else -2 for ret in next_Return]
    classret=pd.DataFrame(classret)
    classret.index=sd_Return.index
    classret.columns=['classret']
    
    todayclassret=[ 2  if today_Return[date]>sd_Return[date]*1.0 else 1 if today_Return[date]>sd_Return[date]*0.05 else 0 if today_Return[date]>sd_Return[date]*-0.05 else -1 if today_Return[date]>sd_Return[date]*-1.0 else -2 for date in sd_Return.index]
    #classret=[ 2  if ret>s2 else 1 if ret>s1 else 0 if ret>s0 else -1 if ret>sn1 else -2 for ret in next_Return]
    todayclassret=pd.DataFrame(todayclassret)
    todayclassret.index=sd_Return.index
    todayclassret.columns=['todayclassret']
    
    next_Return=pd.DataFrame(next_Return)
    today_Return=pd.DataFrame(today_Return)
    next_Return.columns=['next_Return']
    today_Return.columns=['today_Return']
    return today_Return,next_Return,classret,todayclassret


def readdata():
    colum_names = ['ticker', 'date', 'description', 'sector', 'industry', 'raw_s', 's-volume', 's-dispersion', 'raw-s-delta', 'volume-delta', 'center-date', 'center-time', 'center-time-zone']
    df_2015 = pd.read_csv('SPY2015ActFeed.txt', skiprows = 6, sep = '\t', names = colum_names)
    df_2016 = pd.read_csv('SPY2016ActFeed.txt', skiprows = 6, sep = '\t', names = colum_names)
    df_2017 = pd.read_csv('SPY2017ActFeed.txt', skiprows=6, sep = '\t', names = colum_names)
    df_2018 = pd.read_csv('SPY2018ActFeed.txt', skiprows=6, sep = '\t', names = colum_names)
    df_2019 = pd.read_csv('SPY2019ActFeed.txt', skiprows=6, sep = '\t', names = colum_names)
    #aggregating data
    df_temp = df_2015.append(df_2016, ignore_index = True)
    df_temp = df_temp.append(df_2017, ignore_index = True)
    df_temp = df_temp.append(df_2018, ignore_index = True)
    df_aggregate = df_temp.append(df_2019, ignore_index = True)
    df_datetime = df_aggregate['date'].str.split(' ', n = 1, expand = True )
    df_datetime.columns = ['Date', 'Time']
    df = pd.merge(df_aggregate, df_datetime, left_index = True, right_index = True)
    #filtering based on trading hours and excluding weekends
    df = df[(df['Time'] >= '09:30:00') & (df['Time'] <= '16:00:00')]
    #excluding weekends
    #removing empty columns
    df = df.dropna(axis='columns')
    df=df.drop(columns=['ticker','date','description','center-date','center-time','center-time-zone', 'raw-s-delta', 'volume-delta'])
    df["volume_base_s"]=df["raw_s"]/df["s-volume"]
    df["ewm_volume_base_s"] = df.groupby("Date")["volume_base_s"].apply(lambda x: x.ewm(span=390).mean())
    dffinal = df.groupby('Date').last().reset_index()
    dffinal.index=dffinal['Date']
    dffinal["mean_volume_base_s"] = df.groupby("Date")["volume_base_s"].mean()
    dffinal["mean_raw_s"] = df.groupby("Date")["raw_s"].mean()
    dffinal["mean_s_dispersion"] = df.groupby("Date")["s-dispersion"].mean()
    dffinal['volume_base_s_z']=(dffinal['mean_volume_base_s']-dffinal['mean_volume_base_s'].rolling(26).mean())/dffinal['mean_volume_base_s'].rolling(26).std()
    dffinal['s_dispersion_z']=(dffinal['mean_s_dispersion']-dffinal['mean_s_dispersion'].rolling(26).mean())/dffinal['mean_s_dispersion'].rolling(26).std()
    dffinal['raw_s_MACD_ewma12-ewma26'] = dffinal["mean_raw_s"].ewm(span=12).mean() - dffinal["mean_raw_s"].ewm(span=26).mean()
    df1=dffinal.drop(columns=['Date','raw_s','s-volume','s-dispersion','Time','volume_base_s'])
    df1.columns="spy_"+df1.columns

    colum_names = ['ticker', 'date', 'description', 'sector', 'industry', 'raw_s', 's-volume', 's-dispersion', 'raw-s-delta', 'volume-delta', 'center-date', 'center-time', 'center-time-zone']
    df_2015 = pd.read_csv('ES_F2015ActFeed.txt', skiprows = 6, sep = '\t', names = colum_names)
    df_2016 = pd.read_csv('ES_F2016ActFeed.txt', skiprows = 6, sep = '\t', names = colum_names)
    df_2017 = pd.read_csv('ES_F2017ActFeed.txt', skiprows=6, sep = '\t', names = colum_names)
    df_2018 = pd.read_csv('ES_F2018ActFeed.txt', skiprows=6, sep = '\t', names = colum_names)
    df_2019 = pd.read_csv('ES_F2019ActFeed.txt', skiprows=6, sep = '\t', names = colum_names)
    df_temp = df_2015.append(df_2016, ignore_index = True)
    df_temp = df_temp.append(df_2017, ignore_index = True)
    df_temp = df_temp.append(df_2018, ignore_index = True)
    df_aggregate = df_temp.append(df_2019, ignore_index = True)
    df_datetime = df_aggregate['date'].str.split(' ', n = 1, expand = True )
    df_datetime.columns = ['Date', 'Time']
    df = pd.merge(df_aggregate, df_datetime, left_index = True, right_index = True)
    df = df[(df['Time'] >= '09:30:00') & (df['Time'] <= '16:00:00')]
    df = df.dropna(axis='columns')
    df=df.drop(columns=['ticker','date','description','center-date','center-time','center-time-zone', 'raw-s-delta', 'volume-delta'])
    df["volume_base_s"]=df["raw_s"]/df["s-volume"]
    df["ewm_volume_base_s"] = df.groupby("Date")["volume_base_s"].apply(lambda x: x.ewm(span=390).mean())
    dffinal = df.groupby('Date').last().reset_index()
    dffinal.index=dffinal['Date']
    dffinal["mean_volume_base_s"] = df.groupby("Date")["volume_base_s"].mean()
    dffinal["mean_raw_s"] = df.groupby("Date")["raw_s"].mean()
    dffinal["mean_s_dispersion"] = df.groupby("Date")["s-dispersion"].mean()
    dffinal['volume_base_s_delta']=(dffinal['mean_volume_base_s'][1:]-dffinal['mean_volume_base_s'][:-1].values)
    dffinal['s_dispersion_delta']=(dffinal['mean_s_dispersion'][1:]-dffinal['mean_s_dispersion'][:-1].values)
    dffinal['raw_s_MACD_ewma12-ewma26'] = dffinal["mean_raw_s"].ewm(span=12).mean() - dffinal["mean_raw_s"].ewm(span=26).mean()
    df2=dffinal.drop(columns=['Date','raw_s','s-volume','s-dispersion','Time','volume_base_s'])
    df2.columns="future_"+df2.columns
    testdf = pd.concat([df1, df2], axis=1, sort=False)
    today_Return,next_Return,classret,todayclassret=SPY()
    sd_Return=today_Return.iloc[::-1].rolling(250).std().iloc[::-1]
    sd_Return=sd_Return.dropna()
    sd_Return=sd_Return[1:]
    sd_Return
    sd_Return.columns=['sd_Return']
    testdf = pd.concat([testdf, next_Return,today_Return,classret,sd_Return,todayclassret], axis=1, sort=False,join='inner').dropna()
    return testdf

def using_mstats(s):
    return winsorize(s, limits=[0.005, 0.005])

def adf_test(timeseries):
    #print('Results of Augment Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return(dfoutput)

def stationarity(result):
    plist={}
    for col in result:
        if adf_test(result[col])['p-value']<0.05:
            st=True
        else:
            st=False
        plist[col]=st
    return plist

def predict(df,ntrain,ntest):
    print(df.index[len(df)-1])
    X_train=df[0:ntrain]
    X_test=df[ntrain:ntrain+ntest]
    
    y_train=X_train['next_Return']
#    train_y_class=df_train['classret']
    y_test=X_test['next_Return']
#    test_y_class=df_test['classret']
    X_train.drop(['next_Return', 'classret'], axis=1)
    X_test.drop(['next_Return', 'classret'], axis=1)
    X_train=X_train.apply(using_mstats, axis=0)
    maxtrain=X_train.max()
    mintrain=X_train.min()
    for col in X_test:
        X_test[col][X_test[col] < mintrain[col]] = mintrain[col]
        X_test[col][X_test[col] > maxtrain[col]] = maxtrain[col]
        
    slist=(stationarity(X_train))
    
    slist=pd.DataFrame(slist, index=[0])
    factors=[]
    for i in slist.columns:
        if slist[i][0]==1:
            factors.append(i)
    factors.remove("classret")
    factors.remove("next_Return")
    X_train=X_train[factors]
    X_test=X_test[factors]

    # Preprocess / Standardize data
    sc_X = StandardScaler()
    X_train_std = sc_X.fit_transform(X_train)
    X_test_std = sc_X.transform(X_test)
    y_train = np.array(y_train).reshape(-1,1)
    y_test = np.array(y_test).reshape(-1,1)
    
    best_leaf_nodes = 2
    best_n = 100
    
    RFmodel = RandomForestRegressor(n_estimators = best_n,max_leaf_nodes=best_leaf_nodes,n_jobs=-1)
    #RFmodel = RandomForestClassifier(n_estimators = best_n,max_leaf_nodes=best_leaf_nodes,n_jobs=-1)
    RFmodel.fit(X_train_std, y_train)
    # predict on in-sample and oos
    y_train_pred = RFmodel.predict(X_train_std)
    y_test_pred = RFmodel.predict(X_test_std)

    print([df.index[len(df)-1],y_test_pred])
    print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_pred),
        mean_squared_error(y_test, y_test_pred)))

    print('R^2 train: %.3f, test: %.3f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))
    
    #print('accuracy train: %.3f, test: %.3f' % (
    #        accuracy_score(y_train, y_train_pred),
    #        accuracy_score(y_test, y_test_pred)))
    
    print('explanined variance train: %.3f, test: %.3f' % (
            explained_variance_score(y_train, y_train_pred),
            explained_variance_score(y_test, y_test_pred)))
    return [df.index[len(df)-1],y_test_pred]

def position(pred_y):
    index_y=pred_y[0]
    pred_y=pred_y[1]
    f = open("tempsave.csv", "a")
    f.write("\n")
    q1signal=np.quantile(pred_y,0.25)-0.000001
    print('q1:',q1signal)
    lastsignal=pred_y[len(pred_y)-1]
    print('pred',lastsignal)
    def risk(val,q1signal):
        if val >q1signal:
            return 1
        elif val>0:
            return 0.75
        else:
            return 2
    riskToSpy=(len(pred_y)-1)/sum([risk(n,q1signal) for n in pred_y])
    if (lastsignal>q1signal and lastsignal>0.000001):
        f.write(index_y+','+str(riskToSpy))
        f.close()
        return [index_y,riskToSpy]
    elif(lastsignal>0.000001):
        f.write(index_y+','+str(0.75*riskToSpy))
        f.close()
        return [index_y,0.75*riskToSpy]
    elif(lastsignal>0):
        f.write(index_y+','+str(0))
        f.close()
        return [index_y,0]
    else:
        f.write(index_y+','+str(-1))
        f.close()
        return [index_y,-1]

df=readdata()
ntrain=464
ntest=len(df)-ntrain
pred_y=[predict(df[i:ntrain+ntest+i],ntrain,ntest) for i in range(0,len(df)-ntrain,ntest)]
pred_y=pd.DataFrame(pred_y[0][1].T)
pred_y.index=df[ntrain:].index
pred_y.to_csv('pred_y_rf_daily.csv',sep=',')




df=df[598-450-100+1:]
ntrain=450
ntest=100
pos_y=[position(predict(df[i:ntrain+ntest+i],ntrain,ntest)) for i in range(0,len(df)-ntrain-ntest,1)]
pd.DataFrame(pos_y).to_csv('pos_y_rf_daily.csv',sep=',')

#def plot(df,sellpoint,buypoint,start,last):
#    newdf=copy.deepcopy(df)
#    l1,=plt.plot(newdf['Close'][start:last],linewidth=1)
#    plt.legend(handles=[l1],labels=['Close Price'],loc='best')
#    for i in sellpoint:
#        plt.plot(newdf.iloc[i,:].name,newdf['Close'][i],'rs',markersize=1.5)
#        plt.annotate(str(newdf['Close'][i])[:5],xy=(newdf.iloc[i,:].name,newdf['Close'][i]),xytext=(newdf.iloc[i,:].name,newdf['Close'][i]+0.5),weight='ultralight')
#    for i in buypoint:
#        plt.plot(newdf.iloc[i,:].name,newdf['Close'][i],'ks',markersize=1.5)
#        plt.annotate(str(newdf['Close'][i])[:5],xy=(newdf.iloc[i,:].name,newdf['Close'][i]),xytext=(newdf.iloc[i,:].name,newdf['Close'][i]+0.5),weight='ultralight')
#    plt.show()
#
#plot(df,sellpoint,buypoint,250,1000)