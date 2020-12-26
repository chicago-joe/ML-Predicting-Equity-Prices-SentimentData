#
# # --------------------------------------------------------------------------------------------------
# # read in S-Factor Feed Data
#
# def fnLoadSFactorFeed(ticker='SPY'):
#
#     path = '..\\_data\\sFactorFeed\\'
#
#     colNames = ['ticker', 'date', 'raw-s', 'raw-s-mean', 'raw-volatility',
#                 'raw-score', 's', 's-mean', 's-volatility', 's-score',
#                 's-volume', 'sv-mean', 'sv-volatility', 'sv-score',
#                 's-dispersion', 's-buzz', 's-delta',
#                 'center-date', 'center-time', 'center-time-zone']
#
#     df2015 = pd.read_csv(path + '2015\\{}.txt'.format(ticker), skiprows = 4, sep = '\t')
#     df2016 = pd.read_csv(path + '2016\\{}.txt'.format(ticker), skiprows = 4, sep = '\t')
#     df2017 = pd.read_csv(path + '2017\\{}.txt'.format(ticker), skiprows = 4, sep = '\t')
#     df2018 = pd.read_csv(path + '2018\\{}.txt'.format(ticker), skiprows = 4, sep = '\t')
#     df2019 = pd.read_csv(path + '2019\\{}.txt'.format(ticker), skiprows = 4, sep = '\t')
#
#     # aggregating data
#     df_temp = df2015.append(df2016, ignore_index = True)
#     df_temp = df_temp.append(df2017, ignore_index = True)
#     df_temp = df_temp.append(df2018, ignore_index = True)
#     df_temp = df_temp.append(df2019, ignore_index = True)
#
#     df_datetime = df_temp['date'].str.split(' ', n = 1, expand = True)
#     df_datetime.columns = ['Date', 'Time']
#
#     # merge datetime and aggregate dataframe
#     dfAgg = pd.merge(df_temp, df_datetime, left_index = True, right_index = True)
#
#     # filtering based on trading hours and excluding weekends
#     dfAgg['Date'] = pd.to_datetime(dfAgg['Date'])
#
#     dfAgg = dfAgg.loc[(dfAgg['Date'].dt.dayofweek != 5) & (dfAgg['Date'].dt.dayofweek != 6)]
#     dfAgg = dfAgg[(dfAgg['Time'] >= '09:30:00') & (dfAgg['Time'] <= '16:00:00')]
#
#     # exclude weekends and drop empty columns
#     dfAgg = dfAgg.dropna(axis = 'columns')
#     dfAgg = dfAgg.drop(columns = ['ticker', 'date',
#                                   'raw-s', 'raw-s-mean', 'raw-volatility', 'raw-score',
#                                   'center-date', 'center-time', 'center-time-zone'])
#
#     # aggregate by date
#     dfT = dfAgg.groupby('Date').last().reset_index()
#     dfT.index = dfT['Date']
#
#     dfT = dfT.drop(columns = ['Date', 'Time'])
#     dfT.columns = ticker + ':' + dfT.columns
#
#     return dfT
