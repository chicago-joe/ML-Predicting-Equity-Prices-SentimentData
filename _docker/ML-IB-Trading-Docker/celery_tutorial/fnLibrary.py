
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


import mysql.connector
import pytz
from io import StringIO
import os

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

def setLogging(LOGGING_DIRECTORY = os.path.join(os.path.curdir, dt.today().strftime('%Y-%m-%d')), LOG_FILE_NAME = None, level = 'DEBUG'):

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

def setOutputFilePath(OUTPUT_DIRECTORY = os.path.curdir, OUTPUT_SUBDIRECTORY=None, OUTPUT_FILE_NAME=None):

    if not OUTPUT_FILE_NAME:
        OUTPUT_FILE_NAME = os.path.basename(__file__)
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
        'user':              'aschran89',
        'password':          'jlkrg9tdxt5m3dc0',
        'host':              'smadb-mysql-nyc1-75115-do-user-8745596-0.b.db.ondigitalocean.com',
        'port':             '25060',
        # 'client_flags':      [ClientFlag.SSL, ClientFlag.LOCAL_FILES],
        # 'ssl_ca':            os.path.abspath('..\\..\\.\\_auth\\ssl\\server-ca.pem'),
        # 'ssl_cert':          os.path.abspath('..\\..\\.\\_auth\\ssl\\client-cert.pem'),
        # 'ssl_key':           os.path.abspath('..\\..\\.\\_auth\\ssl\\client-key.pem'),
        'autocommit':        True,
        'allow_local_infile':1,
        'database':          '%s' % dbName
}
    # configAdmin = {
    #     'user':              'doadmin',
    #     'password':          'r6raohf6jqh5g2yc',
    #     'host':              'smadb-mysql-nyc1-75115-do-user-8745596-0.b.db.ondigitalocean.com',
    #     'port':             '25060',
    #     'client_flags':      [ClientFlag.SSL, ClientFlag.LOCAL_FILES],
    #     'ssl_ca':            os.path.abspath('..\\..\\.\\_auth\\ssl\\server-ca.pem'),
    #     'ssl_cert':          os.path.abspath('..\\..\\.\\_auth\\ssl\\client-cert.pem'),
    #     'ssl_key':           os.path.abspath('..\\..\\.\\_auth\\ssl\\client-key.pem'),
        # 'autocommit':        True,
        # 'allow_local_infile':1,
        # 'database':          '%s' % dbName
# }
    conn = mysql.connector.connect(**config)
    return conn


# --------------------------------------------------------------------------------------------------
# upload to mysql database

def fnUploadSQL(df=None, conn=None, tblName=None, mode='REPLACE', colNames=None, unlinkFile=True):
    from sqlalchemy import create_engine

    engine = create_engine('mysql://aschran89:jlkrg9tdxt5m3dc0@smadb-mysql-nyc1-75115-do-user-8745596-0.b.db.ondigitalocean.com:25060/defaultdb')
    # setLogging(LOG_FILE_NAME = 'upload %s-%s.txt' % (tblName, os.getpid()), level='INFO')

    curTime = dt.time(dt.now()).strftime("%H_%M_%S")
    # tmpFile = setOutputFilePath(OUTPUT_SUBDIRECTORY = 'upload', OUTPUT_FILE_NAME = '%s %s-%s.txt' % (tblName, curTime, os.getpid()))
    c = conn.cursor()

    # logging.info("Creating temp file: %s" % tmpFile)
    colsSQL = pd.read_sql('SELECT * FROM %s LIMIT 0;' % (tblName), conn).columns.tolist()

    if tblName == 'tblactivityfeedsma':
        for row in df[colsSQL].itertuples():
            q = """
                    REPLACE INTO %s 
                    VALUES (
                                '%s', '%s', '%s', '%s',
                                '%s', '%s', '%s', 
                                '%s', '%s', '%s', '%s', '%s',
                                '%s', '%s', '%s'
                    )
                """ % (tblName,
                       row.date, row.timestampET, row.ticker_at, row.ticker.tk,
                       row.description, row.sector, row.industry,
                       row._6, row._7, row._8, row._9, row._10,
                       row._11, row._12, row._13)
            c.execute(q)


    if tblName == 'tbllivepredictionfeatures':
        for row in df[colsSQL].itertuples():
            q = """
                    REPLACE INTO %s 
                    VALUES (
                       '%s', '%s', '%s', 
                       '%s' , '%s' , '%s' , '%s' , '%s' , 
                       '%s' , '%s' , '%s' , '%s' , '%s' , 
                       '%s' , '%s' , '%s' , '%s' , '%s' , 
                       '%s' , '%s' , '%s' , '%s' , '%s' , 
                       '%s' , '%s' , '%s' , '%s' , '%s' , '%s' , 
                       '%s' , '%s' , '%s' , '%s' , 
                       '%s' , '%s' , '%s' , '%s' , '%s' , 
                       '%s' , '%s' , '%s' , '%s' , '%s' , 
                       '%s' , '%s' , '%s' , '%s' , '%s' 
                    )
                """ % (tblName,
                       row.date, row.ticker_at, row.ticker_tk,
                       row._4, row._5, row._6, row.Week, row._8,
                       row._9, row._10, row._11, row._12, row._13,
                       row._14, row._15, row._16, row._17, row._18,
                       row._19, row._20, row._21, row._22, row._23,
                       row._24, row._25, row._26, row._27, row._28, row._29,
                       row.rtnStdDev, row.rtnTodayToTomorrow, row.rtnTodayToTomorrowClassified, row.VIX_Close,
                       row._34, row._35, row._36, row._37, row._38,
                       row.Year, row.Month, row.Day, row.Dayofweek, row.Dayofyear,
                       row._44, row._45, row._46, row.warning, row.warningSevere)
            c.execute(q)


    if tblName == 'tbllivepositionsignal_v2':
        for row in df[colsSQL].itertuples():
            q = """
                    REPLACE INTO %s 
                    VALUES (
                       '%s', '%s', '%s', 
                       '%s' , '%s' , '%s' , 
                       '%s' ,  
                    )
                """ % (tblName,
                       row.date, row.ticker_at, row.ticker_tk,
                       row.last_signal, row.signal_Q1, row.signal_rolling_Q1,
                       row.position,
                       )
            c.execute(q)

    if tblName == 'tblsecuritypricespolygon':
        for row in df[colsSQL].itertuples():
            q = """
                    REPLACE INTO %s 
                    VALUES (
                       '%s', '%s',  
                       '%s' , '%s' , '%s' , '%s', 
                       '%s' ,  '%s', '%s',
                       '%s',
                    )
                """ % (tblName,
                       row.ticker_at, row.ticker_tk,
                       row.adjOpen, row.adjHigh, row.adjLow, row.adjClose,
                       row.vwap, row.volume, row.n,
                       row.date,
                       )
            c.execute(q)

    if tblName == 'tblsecuritypricesyahoo':
        for row in df[colsSQL].itertuples():
            q = """
                    REPLACE INTO %s 
                    VALUES (
                       '%s', '%s',  
                       '%s', '%s', '%s', '%s',    
                       '%s', '%s',  '%s',
                       '%s',                   
                    )
                """ % (tblName,
                       row.ticker_at, row.ticker_tk,
                       row.adjOpen, row.adjHigh, row.adjLow, row.adjClose,
                       row.volume, row.dividend, row.splitRatio,
                       row.date,
                       )
            c.execute(q)

    return
# --------------------------------------------------------------------------------------------------





    # check columns in db table vs dataframe
    # colsDF = df.columns.tolist()
    # colsDiff = set(colsSQL).symmetric_difference(set(colsDF))

    # if len(colsDiff) > 0:
    #     logging.warning('----- COLUMN MISMATCH WHEN ATTEMPTING TO UPLOAD %s -----' % tblName)
    #     if len(set(colsSQL) - set(colsDF))> 0:
    #         logging.warning('Columns in %s not found in dataframe: %s' % (tblName, list((set(colsSQL) - set(colsDF)))))
    #     if len(set(colsDF) - set(colsSQL))> 0:
    #         logging.warning('Columns in dataframe not found in %s: %s' % (tblName, list((set(colsDF) - set(colsSQL)))))
    #
    # df.replace([-np.inf,np.inf],np.nan,inplace=True)
    # df.to_sql(con=engine, name='tablename_temp', if_exists='replace')
    # connection = con.connect()
    # connection.execute(text("INSERT INTO tablename SELECT * FROM tablename_temp ON DUPLICATE KEY UPDATE tablename.field_to_update=tablename_temp.field_to_update"))
    # connection.execute(text('DROP TABLE tablename_temp '))
    # df[colsSQL].to_sql(name=tblName,con=engine,if_exists='append',index=False)
    # df[colsSQL].to_csv(tmpFile, sep="\t", na_rep="\\N", float_format="%.8g", header=False, index=False, doublequote=False)
    # query = """LOAD DATA LOCAL INFILE '%s' %s INTO TABLE %s LINES TERMINATED BY '\r\n'""" % \
    #         (tmpFile.replace('\\','/'), mode, tblName)

    # logging.debug(query)
    # rv = c.execute(query)
    # logging.info("Number of rows affected: %s" % len(df))

    # if unlinkFile:
    #     os.unlink(tmpFile)
    #     logging.info("Deleting temporary file: {}".format(tmpFile))
    # logging.info("DONE")
    # return rv
