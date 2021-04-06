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

app = Celery('celery_tutorial', broker=os.environ['BROKER'], backend='rpc://',include=['celery_tutorial.ibClient.uploadSMAdata',
                                                                                       'celery_tutorial.ibClient.models.mlLivePredictionsAlgo',
                                                                                       'celery_tutorial.ibClient.main'])


# --------------------------------------------------------------------------------------------------
# connect to db

connTrading = fnOdbcConnect('defaultdb')

# hour = 16
# minute = 46

CELERYBEAT_SCHEDULE = {
    'fnUploadData': {
        'task': 'celery_tutorial.ibClient.uploadSMAdata.fnUploadSMA',
        'schedule': crontab(hour =15,minute=1,day_of_week = '1,2,3,4,5'),
        # 'schedule': crontab(hour =hour,minute=minute,day_of_week = '1,2,3,4,5'),
        'args': ()
    },
    'fnRandomForestPredictor': {
        'task': 'celery_tutorial.ibClient.models.mlLivePredictionsAlgo.fnLivePredict',
        'schedule': crontab(hour =15,minute=6,day_of_week = '1,2,3,4,5'),
        # 'schedule': crontab(hour =hour,minute=minute + 6,day_of_week = '1,2,3,4,5'),
        'args': ()
    },
    'fnIBTrader': {
        'task': 'celery_tutorial.ibClient.main.fnRunIBTrader',
        # 'schedule': crontab(hour =15,minute=11,day_of_week = '1,2,3,4,5'),
        # 'schedule': crontab(hour =hour,minute=minute+11,day_of_week = '1,2,3,4,5'),
        # 'schedule': crontab(hour =hour,minute=minute,day_of_week = '1,2,3,4,5'),
        'schedule': crontab(
                # hour ='*/',
                            minute='*/10',
                day_of_week = '1,2,3,4,5'),
        'args': ()
    },
}

app.conf.update(
        CELERYD_PREFETCH_MULTIPLIER = 1,
        CELERYD_CONCURRENCY = 1,
        CELERY_ACKS_LATE = True,
        C_FORCE_ROOT = True,
        CELERYBEAT_SCHEDULE = CELERYBEAT_SCHEDULE,
        CELERY_TIMEZONE = 'US/Central'
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
