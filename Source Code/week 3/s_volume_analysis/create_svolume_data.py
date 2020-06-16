from readdata import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


# def maketable(data):

#     tickertable = []
#     ticker_name = []
#     for ticker in data:
#         ticker_name.append(ticker)
#         singleticker=pd.DataFrame.from_dict(data[ticker])
#         singleticker=singleticker.transpose()
#         col_names=('raw_s','raw_s_mean','raw_volatility','raw_score','s','s_mean','s_volatility',
#                    's_score','s_volume','sv_mean','sv_volatility','sv_score','s_dispersion','s_buzz','s_delta','date')

#         singleticker.columns=col_names
#         for names in col_names:
#             if names!= 'date':
#                 singleticker[names] = singleticker[names].astype(float)
#         tickertable.append(singleticker['s_volume'].mean())

#     return tickertable, ticker_name

# def svol_over_time(data):

#     tickertable = []
#     ticker_name = []
#     for ticker in data:
#         ticker_name.append(ticker)
#         singleticker=pd.DataFrame.from_dict(data[ticker])
#         singleticker=singleticker.transpose()
#         col_names=('raw_s','raw_s_mean','raw_volatility','raw_score','s','s_mean','s_volatility',
#                    's_score','s_volume','sv_mean','sv_volatility','sv_score','s_dispersion','s_buzz','s_delta','date')

#         singleticker.columns=col_names
#         for names in col_names:
#             if names!= 'date':
#                 singleticker[names] = singleticker[names].astype(float)
#         tickertable.append(singleticker['s_volume'])

#     return tickertable, ticker_name


# ## convert mean of s_volume table to dataframe
# alldatamean,all_tickers = maketable(df_alldata)

# df_all_data = pd.DataFrame(alldatamean,all_tickers)
# top10_idx = np.argsort(df_all_data)[-2:]
# top10_values = [df_all_data[i] for i in top10_idx]

# # make data frame then write to csv
# df = pd.DataFrame(top10_values)
# df.to_csv('mean')


########## Now check out the 12 tickers change in s_volume over time ############
# df_12ticker_table = []
# ticker_mean,ticker_list = svol_over_time(df_12ticker)
# df_12ticker_data = pd.DataFrame(df_12ticker)


col_names = ('raw_s','raw_s_mean','raw_volatility','raw_score','s','s_mean','s_volatility',
                   's_score','s_volume','sv_mean','sv_volatility','sv_score','s_dispersion','s_buzz','s_delta','date','percent_change')

XLK = df_12ticker_data['XLK']
XLV = df_12ticker_data['XLV']
XLF = df_12ticker_data['XLF']
XLY = df_12ticker_data['XLY']
XLI = df_12ticker_data['XLI']
XLP = df_12ticker_data['XLP']
XLE = df_12ticker_data['XLE']
XLU = df_12ticker_data['XLU']
VNQ = df_12ticker_data['VNQ']
GDX = df_12ticker_data['VOX']
VOX = df_12ticker_data['GDX']

# # write tickers to csv
# with pd.ExcelWriter('output.csv') as writer:
#     XLK.to_excel(writer,sheet_name = 'XLK')
#     XLF.to_excel(writer,sheet_name = 'XLF')
#     XLY.to_excel(writer,sheet_name = 'XLY')
#     XLI.to_excel(writer,sheet_name = 'XLI')
#     XLP.to_excel(writer,sheet_name = 'XLP')
#     XLE.to_excel(writer,sheet_name = 'XLE')
#     XLU.to_excel(writer,sheet_name = 'XLU')
#     VNQ.to_excel(writer,sheet_name = 'VNQ')
#     VOX.to_excel(writer,sheet_name = 'VOX')
#     GDX.to_excel(writer,sheet_name = 'GDX')


## open output.xlsx and import dataframes for percent_change
df_xlk = pd.read_excel('output.xlsx','XLK',index_col=0,usecols=[0,17])
df_xlf = pd.read_excel('output.xlsx','XLF',index_col=0,usecols=[0,17])
df_xly = pd.read_excel('output.xlsx','XLY',index_col=0,usecols=[0,17])
df_xli = pd.read_excel('output.xlsx','XLI',index_col=0,usecols=[0,17])
df_xlp = pd.read_excel('output.xlsx','XLP',index_col=0,usecols=[0,17])
df_xle = pd.read_excel('output.xlsx','XLE',index_col=0,usecols=[0,17])
df_xlu = pd.read_excel('output.xlsx','XLU',index_col=0,usecols=[0,17])
df_vnq = pd.read_excel('output.xlsx','VNQ',index_col=0,usecols=[0,17])
df_gdx = pd.read_excel('output.xlsx','GDX',index_col=0,usecols=[0,17])



# convert SPY to dataframe and output s-metrics to excel
col_names = ('raw_s', 'raw_s_mean', 'raw_volatility', 'raw_score', 's', 's_mean', 's_volatility', 's_score', 's_volume', 'sv_mean', 'sv_volatility', 'sv_score',
             's_dispersion', 's_buzz', 's_delta', 'raw_s_wave', 'raw_s_skew', 'raw_s_kurt', 'raw_score_wave', 'raw_score_skew', 'raw_score_kurt', 's_wave', 's_skew', 's_kurt', 'date')
df = pd.DataFrame(df)
SPY = df['SPY']
SPY.columns = col_names

with pd.ExcelWriter('spy_raw.xlsx') as writer:
    SPY.to_excel(writer, sheet_name='SPY')
