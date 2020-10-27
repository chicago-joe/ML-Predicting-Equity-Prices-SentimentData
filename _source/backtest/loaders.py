import time, os, sys
import pandas as pd
import pandas_market_calendars as mcal
import pandas_datareader as pdr
from glob import glob
import yfinance as yf
from pandas.io.json import json_normalize

import bs4 as bs
import urllib.request
from collections import OrderedDict
from datetime import datetime as dt, timedelta

from tiingo import TiingoClient
tiingo_api_key = '473f1019b1f05c17a44ac39484a1ad8129d597ac'


# --------------------------------------------------------------------------------------------------
# scrape cboe put-call ratio data

def fnGetEquityPCR(ticker='EQUITY', endDate=None):

    # check if data already exists
    fpath = "..\\_data\\equitypcr.csv"

    if os.path.isfile(fpath):
        df = pd.read_csv(fpath, index_col=0, parse_dates=True)
    else:
        # download historical equity pcr
        url = "http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/equitypc.csv"
        df = pd.read_csv(url, index_col=0, parse_dates=True, skiprows=2)
        df.rename(columns={'CALL':'cVol', 'PUT':'pVol', 'TOTAL':'tVol', 'P/C Ratio':'pcRatio'}, inplace=True)

    startDate = pd.to_datetime(df.index[-1]) + pd.to_timedelta(1, unit = 'D')
    if not endDate:
        endDate = dt.today().strftime('%Y-%m-%d')
    else:
        endDate = endDate


    # update data if necessary
    if startDate < endDate:
        # get cbot calendar and create list of trading days
        cbot = mcal.get_calendar('CBOT')

        startDate = cbot.schedule(start_date=startDate, end_date=endDate)
        lstTrdDays = mcal.date_range(startDate, frequency='1D')
        lstTrdDays = pd.Series(pd.to_datetime(lstTrdDays.date, format='%Y-%m-%d'))
        dfP = pd.DataFrame(index=lstTrdDays,columns=['pcr','pVol','cVol','tVol'])

        # format date string
        dfP.index = dfP.index.strftime('%Y-%m-%d')
        dfP = dfP.loc[dfP.index>'2019-10-04']


        # --------------------------------------------------------------------------------------------------
        # user input catching
        data = OrderedDict()
        while True:
            try:
                if ticker.upper() == 'SPX':
                    tableNum = 6
                elif ticker.upper() == 'EQUITY':
                    tableNum = 4
                else:
                    ticker = input("PLEASE INPUT A VALID CBOE TICKER: ").upper()
                if ticker in ('SPX', 'EQUITY'):
                    break
            except ValueError:
                continue


        # loop through trade days
        for idx in dfP.index:

            data[idx] = pd.DataFrame(pd.read_html(
                "https://markets.cboe.com/us/options/market_statistics/daily/?mkt=cone&dt=%s" % idx
            )[0].set_index('RATIOS').rename(columns={'Unnamed: 1': 'pcr'}))

            data[idx] = data[idx].loc[data[idx].index.str.contains('{}'.format(ticker))]
            data[idx].index = ['{}'.format(idx)]
            data[idx].index.name = 'Date'

            tmp = pd.read_html(
                    "https://markets.cboe.com/us/options/market_statistics/daily/?mkt=cone&dt=%s" % idx
                    , skiprows = 1, index_col = 0)[tableNum].drop('OPEN INTEREST').rename(
                    columns = { 'CALL':'cVol', 'PUT':'pVol', 'TOTAL':'tVol' })

            tmp.index = ['{}'.format(idx)]
            tmp.index.name = 'Date'

            data[idx] = data[idx].merge(tmp[['pVol', 'cVol', 'tVol']],
                                        left_index=True,
                                        right_index=True)
            dfP.update(data[idx])
            print('{} pcRatio loaded..'.format(idx))
            time.sleep(3)

        dfP.index.name = 'Date'
        dfP.rename(columns={'pcr':'pcRatio'},inplace=True)
        cols = ['cVol', 'pVol', 'tVol', 'pcRatio']
        dfP = dfP[cols]

        # calculate pcRatio
        dfP['pcRatio'] = dfP['pVol'] / dfP['cVol']
        dfP['pcRatio']= dfP['pcRatio'].astype(float).round(2)
        dfP.index = pd.to_datetime(dfP.index)

        # append new data
        df = df.append(dfP)
        df.index.name = 'Date'
        df.to_csv(fpath)

    return df


# --------------------------------------------------------------------------------------------------
# download VIX data from CBOE

def fnDownloadVIXData(ticker='VIX', startDate=None, endDate=None):

    url = 'http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/vixcurrent.csv'
    df = pd.read_csv(url,skiprows=1,index_col=0)

    # format date string
    df.index = df.index.to_datetime().strftime('%Y-%m-%d')
    return df


# --------------------------------------------------------------------------------------------------
# download EOD stock price data for all tickers

def fnLoadStockPriceData(ticker, startDate=None, endDate=None, source='yahoo'):

    # if type(tickers) != type([]):
    #     lstTickers = [tickers]
    # else:
    #     lstTickers = tickers

    if not startDate:
        startDate = '2020-06-01'
    if not endDate:
        endDate = (dt.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    if not source:
        source = 'yahoo'

    # get list of trading dates from pandas market calendar
    nyse = mcal.get_calendar('NYSE')
    nyseCal = nyse.schedule(start_date=startDate,end_date=endDate,tz=None)
    lstTrdDays = nyseCal.index.strftime('%Y-%m-%d').to_list()

    cols = ['adjClose', 'adjVolume']

    if source == 'tiingo':

        config = { }
        config['session'] = True
        config['api_key'] = tiingo_api_key
        client = TiingoClient(config)

        dataStk = client.get_ticker_price(ticker,
                                          startDate = startDate,
                                          endDate = endDate,
                                          fmt = 'json',
                                          frequency = 'daily')
        df = json_normalize(dataStk).set_index('date')
        df.index = pd.to_datetime(df.index).tz_localize(None).strftime('%Y-%m-%d')

    elif source == 'yahoo':
    ## download EOD price data from yahoo finance
        # df = pd.DataFrame()
        dataStk = yf.download(ticker,
                              start = startDate,
                              end = endDate,
                              actions = True,
                              auto_adjust=True)
        print('%s loaded...' % ticker)

        df = dataStk.rename(columns={'Adj Close':'adjClose', 'Volume':'adjVolume'})
        df.index.name = 'date'

    return df[cols]


# --------------------------------------------------------------------------------------------------
# run main

if __name__ == '__main__':

    # total index pcr
    # url = "http://www.cboe.com/publish/scheduledtask/mktdata/datahouse/totalpc.csv"
    print('----- END PROGRAM -----')


# --------------------------------------------------------------------------------------------------
#

# sauce = urllib.request.urlopen('https://markets.cboe.com/us/options/market_statistics/daily/?mkt=cone&dt=2020-07-16/').read()
# soup = bs.BeautifulSoup(sauce,'lxml')
# nav = soup.nav
#
# table = soup.find('table')
# table_rows=table.find_all('tr')

# find all data table tags
#
# for tr in table_rows:
#     td = tr.find_all('td')
#     row = [i.text for i in td]
#     print(row)

