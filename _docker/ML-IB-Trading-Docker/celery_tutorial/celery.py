from __future__ import absolute_import, unicode_literals

# from worker import celeryconfig
from celery.schedules import crontab
import time
from datetime import datetime
import psycopg2
import datetime
import os, sys
from datetime import datetime as dt
import pandas as pd
import numpy as np
import logging

from random import randint
from importlib import reload
from ib_insync import *
import mysql.connector

from psycopg2 import OperationalError, errorcodes, errors
import pytz
import psycopg2.extras as extras
from io import StringIO

import os
# sys.path.remove('/opt/project/celery_tutorial')
# sys.path.remove('/opt/project/celery_tutorial')
# sys.path.remove('/opt/project/celery_tutorial/C')
from celery import Celery
# sys.path.append('/opt/project/celery_tutorial')


# set the default Django settings module for the 'celery' program.
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'celery_tutorial.settings')

# app = Celery('celery_tutorial', broker=os.environ['BROKER'], backend='rpc://', include=['worker.tasks'])
app = Celery('celery_tutorial', broker=os.environ['BROKER'], backend='rpc://',)
# app.autodiscover_tasks()


# --------------------------------------------------------------------------------------------------
# fnUploadSQL

# Define a connect function for PostgreSQL database server
def connect(conn_params_dic):
    conn = None
    try:
        print('Connecting to the PostgreSQL...........')
        conn = psycopg2.connect(**conn_params_dic)
        print("Connection successfully..................")

    except OperationalError as err:
        # passing exception to function
        show_psycopg2_exception(err)
        # set the connection to 'None' in case of error
        conn = None
    return conn

# Define a function that handles and parses psycopg2 exceptions
def show_psycopg2_exception(err):
    # get details about the exception
    err_type, err_obj, traceback = sys.exc_info()
    # get the line number when exception occured
    line_n = traceback.tb_lineno
    # print the connect() error
    logging.error("\npsycopg2 ERROR:", err, "on line number:", line_n)
    logging.error("psycopg2 traceback:", traceback, "-- type:", err_type)
    # psycopg2 extensions.Diagnostics object attribute
    logging.debug("\nextensions.Diagnostics:", err.diag)
    # print the pgcode and pgerror exceptions
    logging.debug("pgerror:", err.pgerror)
    logging.debug("pgcode:", err.pgcode, "\n")



# --------------------------------------------------------------------------------------------------
# pandas settings

def setPandas():

    import warnings
    warnings.simplefilter('ignore', category=FutureWarning)

    options = {
        'display': {
            'max_columns': None,
            'max_colwidth': 800,
            'colheader_justify': 'center',
            'max_rows': 30,
            # 'min_rows': 10,
            'precision': 5,
            'float_format': '{:,.5f}'.format,
            # 'max_seq_items': 50,         # Max length of printed sequence
            'expand_frame_repr': False,  # Don't wrap to multiple pages
            # 'show_dimensions': False
        },
        'mode': {
            'chained_assignment': None   # Controls SettingWithCopyWarning
        }
    }
    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(f'{category}.{op}', value)  # Python 3.6+

    return

# --------------------------------------------------------------------------------------------------
# set up logging

def setLogging(LOGGING_DIRECTORY = os.path.join('..\\..\\logging\\', dt.today().strftime('%Y-%m-%d')), LOG_FILE_NAME = None, level = 'INFO'):

    # reloads logging (useful for iPython only)
    reload(logging)

    LOG_FILE_NAME = LOG_FILE_NAME.replace('.py','.log')

    # init logging
    handlers = [logging.StreamHandler(sys.stdout)]

    if not os.path.exists(LOGGING_DIRECTORY):
        os.makedirs(LOGGING_DIRECTORY)
    handlers.append(logging.FileHandler(os.path.join(LOGGING_DIRECTORY, LOG_FILE_NAME), 'a'))

    # noinspection PyArgumentList
    logging.basicConfig(level = level,
                        format = '%(asctime)s - %(levelname)s - %(message)s',
                        datefmt = '%m/%d/%Y %I:%M:%S %p',
                        handlers = handlers)
    return


# --------------------------------------------------------------------------------------------------
# set up output filepath

def setOutputFilePath(
        OUTPUT_DIRECTORY = os.path.curdir,
        OUTPUT_SUBDIRECTORY=None,
        OUTPUT_FILE_NAME=None):

    if not OUTPUT_FILE_NAME:
        OUTPUT_FILE_NAME = ''
    else:
        OUTPUT_FILE_NAME = OUTPUT_FILE_NAME

    if OUTPUT_SUBDIRECTORY:
        OUTPUT_DIRECTORY = os.path.join(OUTPUT_DIRECTORY, OUTPUT_SUBDIRECTORY)
        if not os.path.exists(OUTPUT_DIRECTORY):
            os.makedirs(OUTPUT_DIRECTORY)
    else:
        if not os.path.exists(OUTPUT_DIRECTORY):
            os.makedirs(OUTPUT_DIRECTORY)

    path = os.path.join(OUTPUT_DIRECTORY, OUTPUT_FILE_NAME)
    logging.info('%s saved in: %s' % (OUTPUT_FILE_NAME, OUTPUT_DIRECTORY))

    return path


# --------------------------------------------------------------------------------------------------
# connect to mysql database

def fnOdbcConnect(dbName='smadb'):

    config = {
        'user':              'root',
        # 'password':          'Quant1984!',
        'password':          'trading',
        'host':              'mysql',
        'port':             '3306',
        # 'client_flags':      [ClientFlag.SSL, ClientFlag.LOCAL_FILES],
        # 'ssl_ca':            os.path.abspath('..\\..\\.\\_auth\\ssl\\server-ca.pem'),
        # 'ssl_cert':          os.path.abspath('..\\..\\.\\_auth\\ssl\\client-cert.pem'),
        # 'ssl_key':           os.path.abspath('..\\..\\.\\_auth\\ssl\\client-key.pem'),
        'autocommit':        True,
        'allow_local_infile':1,
        'database':          '%s' % dbName
}

    conn = mysql.connector.connect(**config)

    return conn



def fnUploadMySQL(df=None, conn=None, tblName=None, mode='REPLACE', colNames=None, unlinkFile=True):

    setLogging(LOG_FILE_NAME = 'upload %s-%s.txt' % (tblName, os.getpid()), level='INFO')

    curTime = dt.time(dt.now()).strftime("%H_%M_%S")
    tmpFile = setOutputFilePath(OUTPUT_SUBDIRECTORY = 'upload', OUTPUT_FILE_NAME = '%s %s-%s.txt' % (tblName, curTime, os.getpid()))
    c = conn.cursor()

    logging.info("Creating temp file: %s" % tmpFile)
    colsSQL = pd.read_sql('SELECT * FROM %s LIMIT 0;' % (tblName), conn).columns.tolist()

    if colNames:
        # check columns in db table vs dataframe
        colsDF = df[colNames].columns.tolist()
        colsDiff = set(colsSQL).symmetric_difference(set(colsDF))

        if len(colsDiff) > 0:
            logging.warning('----- COLUMN MISMATCH WHEN ATTEMPTING TO UPLOAD %s -----' % tblName)
            if len(set(colsDF) - set(colsSQL))> 0:
                logging.warning('Columns in dataframe not found in %s: \n%s' % (tblName, list((set(colsDF) - set(colsSQL)))))
            else:
                df[colsDF].to_csv(tmpFile, sep="\t", na_rep="\\N", float_format="%.8g", header=False, index=False, doublequote=False)
                query = """LOAD DATA LOCAL INFILE '%s' %s INTO TABLE %s LINES TERMINATED BY '\n'(%s)""" % \
                        (tmpFile.replace('\\','/'), mode, tblName, colsDF)

                logging.debug(query)
                rv = c.execute(query)
                logging.debug("Number of rows affected: %s" % len(df))
                return rv

    # check columns in db table vs dataframe
    colsDF = df.columns.tolist()
    colsDiff = set(colsSQL).symmetric_difference(set(colsDF))

    if len(colsDiff) > 0:
        logging.warning('----- COLUMN MISMATCH WHEN ATTEMPTING TO UPLOAD %s -----' % tblName)
        if len(set(colsSQL) - set(colsDF))> 0:
            logging.warning('Columns in %s not found in dataframe: %s' % (tblName, list((set(colsSQL) - set(colsDF)))))
        if len(set(colsDF) - set(colsSQL))> 0:
            logging.warning('Columns in dataframe not found in %s: %s' % (tblName, list((set(colsDF) - set(colsSQL)))))


    df[colsSQL].to_csv(tmpFile, sep="\t", na_rep="\\N", float_format="%.8g", header=False, index=False, doublequote=False)
    query = """LOAD DATA LOCAL INFILE '%s' %s INTO TABLE %s LINES TERMINATED BY '\n'""" % \
            (tmpFile.replace('\\','/'), mode, tblName)

    logging.debug(query)
    rv = c.execute(query)
    logging.info("Number of rows affected: %s" % len(df))

    # if (unlinkFile.lower() == 'yes') | (unlinkFile.lower() == 'y'):
    if unlinkFile:
        os.unlink(tmpFile)
        logging.info("Deleting temporary file: {}".format(tmpFile))
    logging.info("DONE")

    return rv





connTrading = fnOdbcConnect('trading')
# connParams = {'dbname':'trading','user':'trading','host':'postgres','password':'trading'}
# connTrading = connect(connParams)
# connTrading.autocommit=True


# app.conf.update(C_FORCE_ROOT=True, CELERY_WORKER_SEND_TASK_EVENTS = True,
#                 # CELERY_ACCEPT_CONTENT = ['pickle', 'json', 'msgpack', 'yaml'],
#                 CELERY_SEND_EVENTS = True,
# CELERY_SEND_TASK_SENT_EVENT = True,
# # CELERY_IGNORE_RESULT = True,
# # CELERY_DEFAULT_EXCHANGE = 'default',
# CELERY_ACKS_LATE = True,
# CELERYD_PREFETCH_MULTIPLIER = 1,
# CELERY_CREATE_MISSING_QUEUES = True,)

CELERYBEAT_SCHEDULE = {
    'fnUploadData': {
        'task': 'celery_tutorial.celery.load',
        'schedule': 60.0,
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

@app.task(name='add')
def add(x, y):
    print(x+y)
    return x + y


@app.task
def sleep(seconds):
    time.sleep(seconds)


@app.task
def echo(msg, timestamp=False):
    return "%s: %s" % (datetime.now(), msg) if timestamp else msg


@app.task
def error(msg):
    raise Exception(msg)



@app.task
def load(name='load'):
    ib = IB()
    while True:
        if not ib.isConnected():
            try:
                id = randint(0, 9999)
                ib.connect('tws', 4003, clientId=id, timeout=30)
                break
            except Exception as e:
                print(e)
                time.sleep(15)
                continue
        else:
            break

    # ib.connect('tws', 4003, clientId=1)
    contract = Forex('EURUSD')
    bars = ib.reqHistoricalData(contract, endDateTime='', durationStr='600 S',
                            barSizeSetting='1 min', whatToShow='MIDPOINT', useRTH=True)
    bars=util.df(bars)
    # bars.rename(columns={'barCount':'nbars'},inplace=True)
    bars['date'] = pd.to_datetime(bars['date']).dt.tz_localize('UTC').dt.strftime('%Y-%m-%d %H-%M-%S')
    bars = bars[['date','open','high','low','close','volume','average']]
    # bars['date']=bars["date"].apply(lambda x: x.tz_localize('UTC').isoformat())
    fnUploadMySQL(bars, connTrading, 'forex_data_EURUSD', 'REPLACE', None, unlinkFile = True)




def fnUploadSQL(df=None, conn=None, tblName=None, mode='REPLACE', colNames=None, unlinkFile=True):

    setLogging(LOG_FILE_NAME = 'upload %s-%s.txt' % (tblName, os.getpid()), level='INFO')

    curTime = dt.time(dt.now()).strftime("%H_%M_%S")
    tmpFile = setOutputFilePath(OUTPUT_SUBDIRECTORY = 'upload', OUTPUT_FILE_NAME = '%s %s-%s.txt' % (tblName, curTime, os.getpid()))
    c = conn.cursor()

    logging.info("Creating temp file: %s" % tmpFile)
    colsSQL = pd.read_sql('SELECT * FROM %s LIMIT 0;' % (tblName), conn).columns.tolist()

    if colNames:
        # check columns in db table vs dataframe
        colsDF = df[colNames].columns.tolist()
        colsDiff = set(colsSQL).symmetric_difference(set(colsDF))

        if len(colsDiff) > 0:
            logging.warning('----- COLUMN MISMATCH WHEN ATTEMPTING TO UPLOAD %s -----' % tblName)
            if len(set(colsDF) - set(colsSQL))> 0:
                logging.warning('Columns in dataframe not found in %s: \n%s' % (tblName, list((set(colsDF) - set(colsSQL)))))
            else:
                df[colsDF].to_csv(tmpFile, sep="\t", na_rep="\\N", float_format="%.8g", header=False, index=False, doublequote=False)
                try:
                    c.copy_from(open(tmpFile), tblName, columns = colsDF)
                    logging.debug('Data inserted successfully...')
                except (Exception, psycopg2.DatabaseError) as err:
                    # os.remove(tmp_df)
                    # pass exception to function
                    show_psycopg2_exception(err)
                    c.close()
                # query = """COPY '%s' FROM '%s' (DELIMITER('|')); '%s' %s INTO TABLE %s LINES TERMINATED BY '\r\n' (%s)""" % \
                #         (tmpFile.replace('\\','/'), mode, tblName, colsDF)
                # c.copy_from(tmpFile.replace'\\')

                # logging.debug(query)
                # rv = c.execute(query)
                logging.debug("Number of rows affected: %s" % len(df))
                return

    # check columns in db table vs dataframe
    colsDF = df.columns.tolist()
    colsDiff = set(colsSQL).symmetric_difference(set(colsDF))

    if len(colsDiff) > 0:
        logging.warning('----- COLUMN MISMATCH WHEN ATTEMPTING TO UPLOAD %s -----' % tblName)
        if len(set(colsSQL) - set(colsDF))> 0:
            logging.warning('Columns in %s not found in dataframe: %s' % (tblName, list((set(colsSQL) - set(colsDF)))))
        if len(set(colsDF) - set(colsSQL))> 0:
            logging.warning('Columns in dataframe not found in %s: %s' % (tblName, list((set(colsDF) - set(colsSQL)))))


    df[colsSQL].to_csv(tmpFile, sep="\t", na_rep="\\N", float_format="%.8g", header=False, index=False, doublequote=False)
    # f = open(tmpFile.replace('\\','/'), 'r')
    try:
        c.copy_from(open(tmpFile), tblName)
        logging.debug('Data inserted successfully...')
    except (Exception, psycopg2.DatabaseError) as err:
        # os.remove(tmp_df)
        # pass exception to function
        show_psycopg2_exception(err)
        c.close()


    # query = """
    #             COPY %s
    #             FROM '%s'
    #             (DELIMITER('|'))
    #             ;
    #             """ % (tblName, tmpFile.replace('\\', '/'), mode)

    # logging.debug(query)
    # rv = c.execute(query)
    logging.info("Number of rows affected: %s" % len(df))

    # if (unlinkFile.lower() == 'yes') | (unlinkFile.lower() == 'y'):
    if unlinkFile:
        os.unlink(tmpFile)
        logging.info("Deleting temporary file: {}".format(tmpFile))
    logging.info("DONE")
    return










# app.conf.update(
# CELERYD_PREFETCH_MULTIPLIER=1,
# CELERYD_CONCURRENCY=1,
# CELERY_ACKS_LATE=True,
# CELERY_TIMEZONE = 'US/Central'
# )

#
#
# @app.task
# def add(x, y):
#     return x + y
#
# @app.task
# def mul(x, y):
#     return x * y
#
# @app.task
# def xsum(numbers):
#     return sum(numbers)

# @app.on_after_configure.finalize
# def setup_periodic_tasks(sender, **kwargs):
#     # Calls test('hello') every 10 seconds.
#     sender.add_periodic_task(10.0, test.s('hello'), name='add every 10')
#
#     # Calls test('world') every 30 seconds
#     # sender.add_periodic_task(30.0, test.s('world'), expires=10)
#
#     # Executes every Monday morning at 7:30 a.m.
#     # sender.add_periodic_task(
#     #     crontab(hour=7, minute=30, day_of_week=1),
#     #     test.s('Happy Mondays!'),
#     # )

# @app.task
# def test(arg):
#     print(arg)
# #

# app.conf.update(
# CELERYD_PREFETCH_MULTIPLIER=1,
# CELERYD_CONCURRENCY=1,
# CELERY_ACKS_LATE=True,
# CELERY_TIMEZONE = 'US/Central'
# )

# @app.task
# def add(x, y):
#     return x + y
#
# @app.task
# def mul(x, y):
#     return x * y
#
# @app.task
# def xsum(numbers):
#     return sum(numbers)
#
# @app.on_after_configure.finalize
# def setup_periodic_tasks(sender, **kwargs):
#     # Calls test('hello') every 10 seconds.
#     sender.add_periodic_task(10.0, test.s('hello'), name='add every 10')
#
#     # Calls test('world') every 30 seconds
#     # sender.add_periodic_task(30.0, test.s('world'), expires=10)
#
#     # Executes every Monday morning at 7:30 a.m.
#     # sender.add_periodic_task(
#     #     crontab(hour=7, minute=30, day_of_week=1),
#     #     test.s('Happy Mondays!'),
#     # )
# #
# @app.task
# def test(arg):
#     print(arg)
# #


# app.conf.update( CELERY_WORKER_SEND_TASK_EVENTS = True,
#                 CELERY_ACCEPT_CONTENT = ['pickle', 'json', 'msgpack', 'yaml'],
                # CELERY_SEND_EVENTS = True,
# CELERY_SEND_TASK_SENT_EVENT = True,
# CELERY_IGNORE_RESULT = True,
# CELERY_DEFAULT_EXCHANGE = 'default',
# CELERY_ACKS_LATE = True,
# CELERYD_PREFETCH_MULTIPLIER = 1,
# CELERY_CREATE_MISSING_QUEUES = True,)