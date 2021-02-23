
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
    conn = mysql.connector.connect(**config)
    return conn


# --------------------------------------------------------------------------------------------------
# upload to mysql database

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
            if len(set(colsDF) - set(colsSQL)) > 0:
                logging.warning('Columns in dataframe not found in %s: \n%s' % (tblName, list((set(colsDF) - set(colsSQL)))))
            else:
                df[colsDF].to_csv(tmpFile, sep="\t", na_rep="\\N", float_format="%.8g", header=False, index=False, doublequote=False)
                query = """LOAD DATA LOCAL INFILE '%s' %s INTO TABLE %s LINES TERMINATED BY '\r\n'(%s)""" % \
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
    query = """LOAD DATA LOCAL INFILE '%s' %s INTO TABLE %s LINES TERMINATED BY '\r\n'""" % \
            (tmpFile.replace('\\','/'), mode, tblName)

    logging.debug(query)
    rv = c.execute(query)
    logging.info("Number of rows affected: %s" % len(df))

    if unlinkFile:
        os.unlink(tmpFile)
        logging.info("Deleting temporary file: {}".format(tmpFile))
    logging.info("DONE")
    return rv

