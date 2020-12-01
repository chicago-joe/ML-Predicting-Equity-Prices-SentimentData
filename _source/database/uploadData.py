# uploadData.py
#
#
# This script uploads data from SMA and other loader functions to MYSQL databases
# --------------------------------------------------------------------------------------------------
# created by joe.loss

from datetime import datetime as dt, timedelta
import yfinance as yf
from yfinance import shared
import glob, os, sys,logging
import pandas as pd
from fnCommon import setPandas, setLogging, setOutputFilePath, fnUploadSQL, fnOdbcConnect
from datetime import datetime as dt
import numpy as np
import json, csv, requests

LOG_FILE_NAME = os.path.basename(__file__)
LOG_LEVEL = 'DEBUG'

dataPath = '.\\_source\\_data\\activityFeed'
# dataPath = 'C:\\Users\\jloss\\Dropbox\\Coding\\SMA Data Upload\\Activity Feed Data\\tar'


# --------------------------------------------------------------------------------------------------
# download stock price data from YAHOO

def fnGetPricesYahoo(tickers, startDate='2015-01-02', endDate=None, freq='daily'):

    lstTickers = pd.Series(tickers).sort_values().reset_index(drop=True).to_list()
    lstTickers = [sym.replace('.', '-') for sym in lstTickers]
    strTickers = " ".join(lstTickers)

    stkData = pd.DataFrame()
    lstErrors=[]

    if not startDate:
        startDate = pd.to_datetime('2015-01-01').strftime('%Y-%m-%d')
    if not endDate:
        endDate = pd.to_datetime(dt.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    if not freq:
        freq = 'daily'

    logging.info('Downloading %s price data from Yahoo. startDate: %s, endDate: %s' % (freq, startDate, endDate))

    try:

        res = yf.download(strTickers,
                          start = startDate,
                          end = endDate,
                          actions = True,
                          group_by = 'ticker',
                          auto_adjust = True,
                          back_adjust = True
                          )
        # pivot into relational dataframe
        t = res.unstack(level = 1).reset_index().rename(columns = { 'level_0':'ticker_tk', 'level_1':'metric', 0:'value' })
        tmp = t.pivot(index = ['ticker_tk', 'Date'], columns = 'metric', values = 'value').reset_index() #.drop(columns = ['Adj Close'])
        stkData = tmp.loc[~tmp.ticker_tk.isin(lstErrors)].fillna(0)

    except Exception as e:
        print(e)
        pass

    # log errors
    dfErr = shared._ERRORS.items()
    for ticker, desc in dfErr:
        lstErrors.append(ticker)
        logging.warning("'{}' failed to download. - {}".format(ticker, desc))


    stkData.rename(columns = {
            'Date':        'date',
            # 'level_1':     'ticker_tk',
            'Open':        'adjOpen',
            'High':        'adjHigh',
            'Low':         'adjLow',
            'Close':       'adjClose',
            'Volume':      'volume',
            'Dividends':   'dividend',
            'Stock Splits':'splitRatio'
    }, inplace = True)

    # map asset types on ticker
    stkData['ticker_tk'] = stkData['ticker_tk'].str.replace('-', '.')

    stkData['ticker_at'] = np.where(stkData['ticker_tk'] != 'ES_F', 'EQT', 'FUT')
    stkData['date'] = pd.to_datetime(stkData['date']).dt.tz_localize(None).dt.strftime('%Y-%m-%d')

    # drop dates with nonexisting adjClose / splitRatio
    stkData = stkData.loc[~((stkData.splitRatio == 0) & (stkData.adjClose == 0))]

    return stkData, dfErr


# --------------------------------------------------------------------------------------------------
# todo:
# check and update data every week
# def fnUpdateTblStockPriceYahoot()
#     # Missing Dates (Rows) if end date 1 wk greater than last date read
#     if (end_date - max(lstFiDates)).days > 7:
#         dictUpdtTick['updtTickers'] = {
#                 'tickers':  [i for i in lstRdTickers if i in tickers],
#                 'startdate':max(lstFiDates),
#                 'enddate':  end_date,
#                 'rUpdate':  'Rows'
#         }
#         dfReadRets = dfReadRets[['creturns-' + s for s in dictUpdtTick['updtTickers']['tickers']]]
#     return


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# run main

if __name__=='__main__':

    setPandas()
    setLogging(LOG_FILE_NAME = LOG_FILE_NAME, level = LOG_LEVEL)
    # setLogging(LOG_FILE_NAME = 'uploadData - tmp.log', level = LOG_LEVEL)


    try:

        # --------------------------------------------------------------------------------------------------
        # iterate through SMA text files and upload data to SMA Activity Feed table

        # dataPath = '..\\.\\_data\\activityFeed'
        files = glob.glob(dataPath + '\\**\\*.txt', recursive = True)

        # # get unique tickers to pull and load stock price data
        serTickers = pd.Series(files).str.rstrip('.txt').str.rsplit('\\', expand = True)[9]
        serTickers.name = 'ticker_tk'
        reqTickers = serTickers.unique().tolist()


        q = """select distinct ticker_tk from smadb.tblactivityfeedsma"""
        conn = fnOdbcConnect('smadb')
        df1 = pd.read_sql_query(q, conn)

        q = """select distinct ticker_tk from smadb.tblsfactorfeedsma"""
        df2 = pd.read_sql_query(q, conn)

        df = df1.append(df2, ignore_index = True)
        df.ticker_tk.unique().tolist()
        reqTickers = df.ticker_tk.unique().tolist()

        conn.disconnect()
        conn.close()

        # # load stock price data
        dfStk, dfStkErr = fnGetPricesYahoo(tickers=reqTickers, startDate = '2010-01-02', endDate=None, freq='daily')

        cols = ['ticker_at', 'ticker_tk', 'date',
                'adjClose', 'dividend', 'adjHigh', 'adjLow',
                'adjOpen', 'splitRatio', 'volume',]

        dfStk = dfStk[cols]

        conn = fnOdbcConnect('smadb')
        fnUploadSQL(dfStk, conn, 'smadb', 'tblsecuritypricesyahoo','REPLACE',None,True)
        conn.close()


        # --------------------------------------------------------------------------------------------------
        # upload sma activity feed data

        total = len(files)
        count = 0
        lstFiles = []

        dfRank = pd.DataFrame()
        dfRank = pd.DataFrame(columns = ['ticker_tk',
                                         'date',
                                         'description',
                                         'sector',
                                         'industry',
                                         'raw-s',
                                         's-volume',
                                         's-dispersion',
                                         'raw-s-delta',
                                         'volume-delta',
                                         'center-date',
                                         'center-time',
                                         'center-time-zone'])

        for file in files:
            count += 1

            df = pd.read_csv(file, skiprows = 5, sep = '\t', dtype={'ticker':'str'})
            df.rename(columns = { 'ticker':'ticker_tk' }, inplace = True)


            if not df.empty:
                ticker = df['ticker_tk'][0]

                if ticker in ['ACWI', 'AFK', 'AGG',
                                                         'AGQ',
                                                         'ALT',
                                                         'AMLP',
                                                         'AND',
                                                         'ARGT',
                                                         'ASHR',
                                                         'AUD',
                                                         'BAL',
                                                         'BBH',
                                                         'BIB',
                                                         'BIL',
                                                         'BIS',
                                                         'BKLN',
                                                         'BND',
                                                         'BNO',
                                                         'BOIL',
                                                         'BRZU',
                                                         'CAD',
                                                         'CNY',
                                                         'COPX',
                                                         'CORN',
                                                         'CURE',
                                                         'CYB',
                                                         'DBA',
                                                         'DBB',
                                                         'DBC',
                                                         'DBO',
                                                         'DGAZ',
                                                         'DGL',
                                                         'DGLD',
                                                         'DGP',
                                                         'DIA',
                                                         'DIG',
                                                         'DNO',
                                                         'DOG',
                                                         'DRN',
                                                         'DRR',
                                                         'DSLV',
                                                         'DTO',
                                                         'DUST',
                                                         'DXJ',
                                                         'EDC',
                                                         'EDV',
                                                         'EDZ',
                                                         'EEM',
                                                         'EFA',
                                                         'ELD',
                                                         'EMB',
                                                         'EMG',
                                                         'EPI',
                                                         'ERO',
                                                         'ERX',
                                                         'ERY',
                                                         'ES_F',
                                                         'EWA',
                                                         'EWG',
                                                         'EWH',
                                                         'EWI',
                                                         'EWJ',
                                                         'EWT',
                                                         'EWU',
                                                         'EWW',
                                                         'EWY',
                                                         'EWZ',
                                                         'EZU',
                                                         'FAN',
                                                         'FAS',
                                                         'FAZ',
                                                         'FBT',
                                                         'FCG',
                                                         'FDN',
                                                         'FEZ',
                                                         'FM',
                                                         'FXI',
                                                         'GASL',
                                                         'GBB',
                                                         'GDX',
                                                         'GDXJ',
                                                         'GLD',
                                                         'GLDI',
                                                         'GLL',
                                                         'GREK',
                                                         'GSG',
                                                         'GXC',
                                                         'HEDJ',
                                                         'HYG',
                                                         'IAU',
                                                         'IBB',
                                                         'IEF',
                                                         'IEI',
                                                         'IEMG',
                                                         'IEV',
                                                         'IGN',
                                                         'IGV',
                                                         'IHI',
                                                         'IJR',
                                                         'ILF',
                                                         'INDA',
                                                         'INDY',
                                                         'INR',
                                                         'IPO',
                                                         'ITB',
                                                         'IVV',
                                                         'IWF',
                                                         'IWM',
                                                         'IYR',
                                                         'JDST',
                                                         'JJC',
                                                         'JNK',
                                                         'JNUG',
                                                         'JO',
                                                         'JYN',
                                                         'KBE',
                                                         'KOL',
                                                         'KOLD',
                                                         'KRE',
                                                         'LIT',
                                                         'LQD',
                                                         'MCHI',
                                                         'MDY',
                                                         'MOO',
                                                         'MORL',
                                                         'MTUM',
                                                         'MUB',
                                                         'NGE',
                                                         'NKY',
                                                         'NOBL',
                                                         'NUGT',
                                                         'OIH',
                                                         'OIL',
                                                         'OLEM',
                                                         'OLO',
                                                         'ONG',
                                                         'PALL',
                                                         'PFF',
                                                         'PGF',
                                                         'PGJ',
                                                         'PIN',
                                                         'PKB',
                                                         'PPLT',
                                                         'PSQ',
                                                         'PST',
                                                         'QID',
                                                         'QLD',
                                                         'QQQ',
                                                         'REM',
                                                         'RSP',
                                                         'RSX',
                                                         'RTH',
                                                         'RUSL',
                                                         'RUSS',
                                                         'RWM',
                                                         'SCO',
                                                         'SDOW',
                                                         'SDS',
                                                         'SGG',
                                                         'SGOL',
                                                         'SH',
                                                         'SHY',
                                                         'SIL',
                                                         'SILJ',
                                                         'SIVR',
                                                         'SLV',
                                                         'SLX',
                                                         'SMH',
                                                         'SOCL',
                                                         'SOXL',
                                                         'SPLV',
                                                         'SPXS',
                                                         'SPXU',
                                                         'SPY',
                                                         'SQQQ',
                                                         'SSO',
                                                         'TAN',
                                                         'TBF',
                                                         'TBT',
                                                         'TIP',
                                                         'TLH',
                                                         'TLT',
                                                         'TMF',
                                                         'TMV',
                                                         'TNA',
                                                         'TQQQ',
                                                         'TTT',
                                                         'TUR',
                                                         'TVIX',
                                                         'TWM',
                                                         'TZA',
                                                         'UCO',
                                                         'UDN',
                                                         'UDOW',
                                                         'UGA',
                                                         'UGAZ',
                                                         'UGL',
                                                         'UGLD',
                                                         'ULE',
                                                         'UNG',
                                                         'UNL',
                                                         'UPRO',
                                                         'URA',
                                                         'URE',
                                                         'USD',
                                                         'USDU',
                                                         'USL',
                                                         'USLV',
                                                         'USO',
                                                         'UST',
                                                         'UUP',
                                                         'UVXY',
                                                         'UWM',
                                                         'UWTI',
                                                         'VDE',
                                                         'VEA',
                                                         'VGK',
                                                         'VGLT',
                                                         'VGT',
                                                         'VIXY',
                                                         'VNQ',
                                                         'VOO',
                                                         'VOX',
                                                         'VT',
                                                         'VTI',
                                                         'VWO',
                                                         'VXX',
                                                         'WEAT',
                                                         'WEET',
                                                         'XAGUSD',
                                                         'XAUUSD',
                                                         'XBI',
                                                         'XES',
                                                         'XHB',
                                                         'XIV',
                                                         'XLB',
                                                         'XLE',
                                                         'XLF',
                                                         'XLI',
                                                         'XLK',
                                                         'XLP',
                                                         'XLU',
                                                         'XLV',
                                                         'XLY',
                                                         'XME',
                                                         'XOP',
                                                         'XRT',
                                                         'YANG',

                                                         'YCS',
                                                            'YINN', 'ZIV', 'ZROZ']:

                    df['ticker_at'] = np.where(df['ticker_tk'].str.endswith('_F'), 'FUT', 'EQT')

                    dfRank = dfRank.append(df, ignore_index = True)
                    pctComplete = count / total
                    logging.debug('{:.2%} percent complete..'.format(pctComplete))


        dfRank['timestampET'] = pd.to_datetime(dfRank['center-date'] + ' ' + dfRank['center-time'])


        # --------------------------------------------------------------------------------------------------
        # drop tickers that aren't full rank:

        nMax = dfRank.groupby('ticker_tk')['center-date'].nunique().max()
        dfRank = dfRank.groupby('ticker_tk')['center-date'].nunique().reset_index()
        dfRank = dfRank.loc[dfRank['center-date'] >= nMax / 2]
        tickersHalfRank = dfRank.ticker_tk.unique().tolist()
        df = df.loc[df.ticker_tk.isin(tickersHalfRank)]

        conn = fnOdbcConnect('smadb')
        frame = pd.concat(lstFiles, axis=0, ignore_index=True)

        fnUploadSQL(df = frame, conn = conn,
                        dbName = 'smadb',
                        tblName = 'tblactivityfeedsma',
                        mode = 'REPLACE',
                        colNames = None,
                        unlinkFile = True)

        conn.disconnect()
        conn.close()


        # --------------------------------------------------------------------------------------------------
        # upload SMA s-factor feed data

        # dataPath = '.\\_source\\_data\\sFactorFeed'

        dataPath = '..\\.\\_data\\sFactorFeed'
        files = glob.glob(dataPath + '\\**\\*.txt', recursive = True)

        total = len(files)
        count = 0
        lstFiles = []

        for file in files:
            df = pd.read_csv(file, skiprows = 4, sep = '\t', dtype={'ticker':'str'})

            if not df.empty:
                df.rename(columns = { 'ticker':'ticker_tk' }, inplace = True)
                df['ticker_at'] = np.where(df['ticker_tk'].str.endswith('_F'), 'FUT', 'EQT')

                lstFiles.append(df)
                count += 1
                pctComplete = count / total
            # logging.debug('{:.2%} percent complete..'.format(pctComplete))


        # --------------------------------------------------------------------------------------------------
        # drop tickers that aren't full rank:

        nMax = df.groupby('ticker_tk')['center-date'].nunique().max()
        dfRank = df.groupby('ticker_tk')['center-date'].nunique().reset_index()
        dfRank = dfRank.loc[dfRank['center-date'] >= nMax / 2]
        tickersHalfRank = dfRank.ticker_tk.unique().tolist()
        df = df.loc[df.ticker_tk.isin(tickersHalfRank)]

        conn = fnOdbcConnect('smadb')
        frame = pd.concat(lstFiles, axis=0, ignore_index=True)

        fnUploadSQL(df = frame, conn = conn,
                        dbName = 'smadb',
                        tblName = 'tblsfactorfeedsma',
                        mode = 'REPLACE',
                        colNames = None,
                        unlinkFile = True)

        conn.disconnect()
        conn.close()


        # --------------------------------------------------------------------------------------------------

        logging.info('----- END PROGRAM -----')

    except Exception as e:
        print(e)


# --------------------------------------------------------------------------------------------------
# polygon minute data

df = pd.DataFrame(index = pd.RangeIndex(0, 500000), columns = ['v', 'vw', 'o', 'c', 'h', 'l', 't', 'n'])
key = 'l_sMUCamAEWkIVdW5znVVzRFgsgZBaez'
r = requests.get('https://api.polygon.io/v2/aggs/ticker/SPY/range/1/minute/2017-12-01/2019-11-02?apiKey=%s&limit=50000' % key)
data = r.json()['results']

df = pd.DataFrame(data).set_index('t')

last = pd.to_datetime(df.index[-1], origin = 'unix', unit = 'ms')
lastDay = last.strftime('%Y-%m-%d')
r = requests.get('https://api.polygon.io/v2/aggs/ticker/SPY/range/1/minute/%s/2019-11-02?apiKey=%s&limit=50000' % (lastDay, key))
data = r.json()['results']
df2 = pd.DataFrame(data).set_index('t')
df = df.append(df2)
df = df.loc[df.index.duplicated() == False]

df.reset_index(inplace = True)
df['ticker_tk'] = 'SPY'
df['ticker_at'] = 'EQT'
df.rename(columns = { 't':'date', 'v':'volume', 'vw':'vwap', 'c':'adjClose', 'h':'adjHigh', 'o':'adjOpen', 'l':'adjLow' }, inplace = True)
df.date = pd.to_datetime(df.date, origin = 'unix', unit = 'ms')
