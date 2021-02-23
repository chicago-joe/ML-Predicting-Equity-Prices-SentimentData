from __future__ import absolute_import, unicode_literals

# import pydevd_pycharm
# pydevd_pycharm.settrace('localhost', port=52792, stdoutToServer=True, stderrToServer=True,suspend=False)

from celery.schedules import crontab
import time
from datetime import datetime
import datetime
import os, sys
from datetime import datetime as dt
import pandas as pd
import numpy as np
import logging
from random import randint
from importlib import reload
from ib_insync import *
import asyncio
import logging
util.logToConsole(logging.INFO)
util.patchAsyncio()
import mysql.connector
import pytz
from io import StringIO
import os

# sys.path.remove('/opt/project/_docker/ML-IB-Trading-Docker/celery_tutorial')
# sys.path.remove('/opt/project/_docker/ML-IB-Trading-Docker/celery_tutorial')
# sys.path.remove('/opt/project/_docker/ML-IB-Trading-Docker/celery_tutorial/C')
from celery import Celery
# sys.path.append('/opt/project/_docker/ML-IB-Trading-Docker/celery_tutorial')

from .fnLibrary import setPandas, fnOdbcConnect, fnUploadSQL
setPandas()

# app = Celery('celery_tutorial', broker=os.environ['BROKER'], backend='rpc://', include=['worker.tasks'])
# app = Celery('celery_tutorial', broker=os.environ['BROKER'], backend='rpc://',include=['celery_tutorial.ibClient.uploadSMAdata'])
app = Celery('celery_tutorial', broker=os.environ['BROKER'], backend='rpc://',)



# --------------------------------------------------------------------------------------------------
# connect to db

connTrading = fnOdbcConnect('defaultdb')

CELERYBEAT_SCHEDULE = {
    'fnUploadData': {
        # 'task': 'celery_tutorial.ibClient.uploadSMAdata',
        'task': 'celery_tutorial.celery.load',
        'schedule': 120.0,
        'args': ()
    },
}

app.conf.update(
    CELERYD_PREFETCH_MULTIPLIER=1,
    CELERYD_CONCURRENCY=1,
    CELERY_ACKS_LATE=True,
        C_FORCE_ROOT=True,
        CELERYBEAT_SCHEDULE=CELERYBEAT_SCHEDULE
)


# --------------------------------------------------------------------------------------------------
# tasks

@app.task
def sleep(seconds):
    time.sleep(seconds)


@app.task
def echo(msg, timestamp=False):
    return "%s: %s" % (datetime.now(), msg) if timestamp else msg


@app.task
def error(msg):
    raise Exception(msg)


ib = IB()

# @app.task(name='load')
@app.task()
def load():
    while True:
        if not ib.isConnected():
            try:
                id = randint(0, 9999)
                ib.connect('tws', 4003, clientId=id, timeout=0)
                ib.sleep(3)
                break
            except Exception as e:
                print(e)
                ib.sleep(3)
                continue

    contract = Forex('EURUSD')
    bars = ib.reqHistoricalData(contract, endDateTime='', durationStr='600 S', barSizeSetting='1 min', whatToShow='MIDPOINT', useRTH=False, keepUpToDate = False)

    bars = util.df(bars)
    bars['date'] = pd.to_datetime(bars['date']).dt.tz_localize('UTC').dt.strftime('%Y-%m-%d %H:%M:%S')
    bars = bars[['date','open','high','low','close','volume','average']]
    fnUploadSQL(bars, connTrading, 'forex_data_EURUSD', 'REPLACE', None, unlinkFile = True)
