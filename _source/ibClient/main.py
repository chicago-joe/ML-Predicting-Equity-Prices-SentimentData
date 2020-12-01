# IE598 Project - Joseph Loss
# An algorithmic trading trading strategy for IBroker's Python API.
#
# Original Templates, Classes, and Parameters located at:
# https://github.com/jamesmawm/High-Frequency-Trading-Model-with-IB
#
# --------------------------------------------------------------------------------------------------
#

from ibClient import HftModel1
import os
from ib_insync import Stock, ContFuture, Future, Forex
from fnCommon import setPandas, fnOdbcConnect, setLogging, fnUploadSQL


# --------------------------------------------------------------------------------------------------
# create IB contract symbiology

def fnCreateIBSymbol(ticker_tk=None, ticker_at=None):
    if not ticker_tk:
        ticker_tk='SPY'
    if not ticker_at:
        ticker_at = 'EQT'

    symIB = None

    if ticker_at == 'EQT':
        symIB = [ticker_tk, Stock(ticker_tk, 'SMART', 'USD')]
    elif ticker_at == 'FUT':
        symIB = [ticker_tk, Future(ticker_tk, 'SMART', 'USD')]

    return symIB


# --------------------------------------------------------------------------------------------------
# read positon / direction from ML model

# def fn

# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# run main

if __name__ == "__main__":

    TWS_HOST = os.environ.get('TWS_HOST', '127.0.0.1')
    TWS_PORT = os.environ.get('TWS_PORT', 7497)

    print('Connecting on host:', TWS_HOST, 'port:', TWS_PORT)

    # init model
    model = HftModel1(
            host = TWS_HOST,
            port = TWS_PORT,
            client_id = 1,
    )
    tickerIB = fnCreateIBSymbol(ticker_tk = 'SPY', ticker_at = 'EQT')

    model.run(to_trade = tickerIB, trade_qty = 100)
