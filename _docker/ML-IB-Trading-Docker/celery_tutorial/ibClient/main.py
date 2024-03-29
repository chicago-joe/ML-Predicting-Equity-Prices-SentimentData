# main.py
#
# Predicting SPY prices using SMA data for IBroker's Python API.
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
from ib_insync import Stock, ContFuture, Future, Forex

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from celery_tutorial.fnLibrary import setPandas, fnOdbcConnect, setLogging, fnUploadSQL

from celery_tutorial.ibClient.models.ibAlgo import HftModel1

# --------------------------------------------------------------------------------------------------
# create IB contract symbiology

def fnCreateIBSymbol(ticker_tk=None, ticker_at=None):

    if not ticker_tk:
        ticker_tk = 'SPY'
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
    if not trdDate:
        trdDate = dt.today().strftime('%Y-%m-%d')


    q = """ 
            SELECT * FROM defaultdb.tbllivepositionsignal_v2
            WHERE ticker_tk = '%s'
            order by date desc
            ;
        """ % ticker_tk

    conn = fnOdbcConnect('defaultdb')
    df = pd.read_sql_query(q, conn)
    conn.close()

    # get position
    if not df.empty:
        df = df.loc[df.date <= pd.to_datetime(trdDate)]

        if len(df) > 1:
            logging.info('Live Position Signal loaded: \n{}'.format(df))
            pos = df['position'].values[0]
            posPr = df['position'].values[1]

        else:
            # todo: check valid date
            logging.error('NO VALID DATE!')
            raise Exception

    else:
        logging.error('NO POSITION LOADED')
        raise Exception

    return pos, posPr


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# run main
from celery_tutorial.celery import app

@app.task
def fnRunIBTrader():
    setPandas()
    setLogging(LOGGING_DIRECTORY = os.path.join('../logging/', dt.today().strftime('%Y-%m-%d')),
               LOG_FILE_NAME = os.path.basename(__file__),
               level = LOG_LEVEL)

    logging.info('Running script {}'.format(os.path.basename(__file__)))
    logging.info('Process ID: {}'.format(os.getpid()))
    curDate = dt.today().strftime('%Y-%m-%d')

    try:

        pos, posPr = fnGetLivePositionSignal(ticker_tk = ticker, trdDate = curDate)


        TWS_HOST = os.environ.get('TWS_HOST', 'tws')
        TWS_PORT = os.environ.get('TWS_PORT', 4003)

        logging.info('Connecting on host: {} port: {}'.format(TWS_HOST, TWS_PORT))

        # init model
        model = HftModel1(
                host = TWS_HOST,
                port = TWS_PORT,
                client_id = 1,
        )

        tickerIB = fnCreateIBSymbol(ticker_tk = 'SPY', ticker_at = 'EQT')
        # tickerIB = fnCreateIBSymbol(ticker_tk = 'EURUSD', ticker_at = 'FX')

        # run model
        model.run(ticker_tk = tickerIB, position = pos, prevPosition = posPr)


        # -------------------------------------------------------------------------------------------------

        logging.info('----- END PROGRAM -----')

    except Exception as e:
        logging.error(str(e), exc_info=True)


    # CLOSE LOGGING
    for handler in logging.root.handlers:
        handler.close()
    logging.shutdown()

# fnRunIBTrader()
