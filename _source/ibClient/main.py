# main.py
#
#
# Predicting SPY prices using SMA data for IBroker's Python API.
#
# Original Templates, Classes, and Parameters located at:
# https://github.com/jamesmawm/High-Frequency-Trading-Model-with-IB
#
# created by joe.loss, November 2020
# --------------------------------------------------------------------------------------------------
# Parameters

ticker = 'SPY'
ticker_at = 'EQT'
LOG_LEVEL = 'INFO'

# --------------------------------------------------------------------------------------------------
# Module Imports

import os, sys, logging
import pandas as pd
import numpy as np
from datetime import datetime as dt
from ibClient import HftModel1
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
    elif ticker_at == 'FX':
        symIB = [ticker_tk, Forex(ticker_tk,'IDEALPRO')]

    return symIB


# --------------------------------------------------------------------------------------------------
# read positon / direction from ML model

def fnGetLivePositionSignal(ticker_tk=None, trdDate=None):

    if not ticker_tk:
        ticker_tk = 'SPY'
    else:
        ticker_tk = ticker_tk
    if not trdDate:
        trdDate = dt.today().strftime('%Y-%m-%d')
    else:
        trdDate = trdDate

    q = """ 
            SELECT * FROM smadb.tbllivepositionsignal
            WHERE ticker_tk='%s'
            AND date = '%s'
            ;
        """ % (ticker_tk, trdDate)

    conn = fnOdbcConnect('smadb')

    df = pd.read_sql_query(q, conn)
    conn.close()

    logging.info('Live Position Signal loaded: \n{}'.format(df))

    # get position
    pos = df['position'].values[0]

    return pos


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# run main

if __name__ == "__main__":

    setPandas()
    setLogging(LOGGING_DIRECTORY = os.path.join('..\\..\\logging\\', dt.today().strftime('%Y-%m-%d')),
               LOG_FILE_NAME = os.path.basename(__file__),
               level = LOG_LEVEL)

    logging.info('Running script {}'.format(os.path.basename(__file__)))
    logging.info('Process ID: {}'.format(os.getpid()))
    curDate = dt.today().strftime('%Y-%m-%d')


    try:

        position = fnGetLivePositionSignal(ticker_tk = ticker, trdDate = curDate)

        TWS_HOST = os.environ.get('TWS_HOST', '127.0.0.1')
        TWS_PORT = os.environ.get('TWS_PORT', 7497)

        logging.info('Connecting on host: {} port: {}'.format(TWS_HOST, TWS_PORT))

        # init model
        model = HftModel1(
                host = TWS_HOST,
                port = TWS_PORT,
                client_id = 1,
        )
        # todo:
        # create IB symbol
        # tickerIB = fnCreateIBSymbol(ticker_tk = ticker, ticker_at = ticker_at)

        tickerIB = fnCreateIBSymbol(ticker_tk = 'EURUSD', ticker_at = 'FX')
        # run model
        model.run(ticker_tk = tickerIB, position = position)


        # -------------------------------------------------------------------------------------------------

        logging.info('----- END PROGRAM -----')

    except Exception as e:
        logging.error(str(e), exc_info=True)

    # CLOSE LOGGING
    for handler in logging.root.handlers:
        handler.close()
    logging.shutdown()
