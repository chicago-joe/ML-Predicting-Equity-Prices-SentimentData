# --------------------------------------------------------------------------------------------------
# created by joe.loss
#
#
# --------------------------------------------------------------------------------------------------
import os, sys
from datetime import datetime as dt
import pandas as pd
import numpy as np
import logging
from importlib import reload


# --------------------------------------------------------------------------------------------------
# pandas settings
import pyodbc


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
# connect to mysql database

def fnOdbcConnect(dsn):
    conn = pyodbc.connect(dsn = dsn, autocommit = 'True')
    return conn


# --------------------------------------------------------------------------------------------------
# fn: Create Option Keys

def fnCreateOptKeys(df, key = 'secKey', has_secType=True):
    optKey = df[[key + '_tk',
                       key + '_yr',
                       key + '_mn',
                       key + '_dy',
                       key + '_xx',
                       key + '_cp']].astype('str').agg(':'.join, axis = 1)

    # return optKey if secType = option, else return stkKey
    if has_secType:
        stkKey = df[key + '_tk'] + ':' + 'Stk'
        key = np.where(df.secType == 'Option', optKey, stkKey)
    else:
        key = optKey

    return key


# --------------------------------------------------------------------------------------------------
# fn: Create accurate fill size +/- for Buy/Sell

def fnCreateBuySellpos(side):
    if side == 'Sell':
        return -1
    else:
        return 1


# --------------------------------------------------------------------------------------------------
# set up logging

def setLogging(LOGGING_DIRECTORY = os.path.join('.\\_source\\_logging', dt.today().strftime('%Y-%m-%d')), LOG_FILE_NAME = None, level = 'INFO'):

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

def setOutputFilePath(OUTPUT_DIRECTORY = os.path.join('.\\results\\'), OUTPUT_SUBDIRECTORY=None, OUTPUT_FILE_NAME=None):

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
# fnUploadSQL


def fnUploadSQL(df=None, conn=None, dbName=None, tblName=None, mode='REPLACE', colNames=None, unlinkFile=True):


    setLogging(LOG_FILE_NAME = 'upload %s.%s-%s.txt' % (dbName, tblName, os.getpid()), level='INFO')

    curTime = dt.time(dt.now()).strftime("%H_%M_%S")
    tmpFile = setOutputFilePath(OUTPUT_SUBDIRECTORY = 'upload', OUTPUT_FILE_NAME = '%s %s-%s.txt' % (tblName, curTime, os.getpid()))

    logging.info("Creating temp file: %s" % tmpFile)
    colsSQL = pd.read_sql('SELECT * FROM %s.%s LIMIT 0;' % (dbName, tblName), conn).columns.tolist()

    if colNames:
        # check columns in db table vs dataframe
        colsDF = df[colNames].columns.tolist()
        colsDiff = set(colsSQL).symmetric_difference(set(colsDF))

        if len(colsDiff) > 0:
            logging.warning('----- COLUMN MISMATCH WHEN ATTEMPTING TO UPLOAD %s -----' % tblName)
            if len(set(colsDF) - set(colsSQL))> 0:
                logging.warning('Columns in dataframe not found in %s.%s: \n%s' % (dbName, tblName, list((set(colsDF) - set(colsSQL)))))
            else:
                df[colsDF].to_csv(tmpFile, sep="\t", na_rep="\\N", float_format="%.8g", header=False, index=False, doublequote=False)
                query = """LOAD DATA LOCAL INFILE '%s' %s INTO TABLE %s.%s LINES TERMINATED BY '\r\n' (%s)""" % \
                        (tmpFile.replace('\\','/'), mode, dbName, tblName, colsDF)

                logging.debug(query)
                rv = conn.execute(query)
                logging.info("Number of rows affected: %s" % len(df))
                return rv

    # check columns in db table vs dataframe
    colsDF = df.columns.tolist()
    colsDiff = set(colsSQL).symmetric_difference(set(colsDF))

    if len(colsDiff) > 0:
        logging.warning('----- COLUMN MISMATCH WHEN ATTEMPTING TO UPLOAD %s -----' % tblName)
        if len(set(colsSQL) - set(colsDF))> 0:
            logging.warning('Columns in %s.%s not found in dataframe: %s' % (dbName, tblName, list((set(colsSQL) - set(colsDF)))))
        if len(set(colsDF) - set(colsSQL))> 0:
            logging.warning('Columns in dataframe not found in %s.%s: %s' % (dbName, tblName, list((set(colsDF) - set(colsSQL)))))


    df[colsSQL].to_csv(tmpFile, sep="\t", na_rep="\\N", float_format="%.8g", header=False, index=False, doublequote=False)
    query = """LOAD DATA LOCAL INFILE '%s' %s INTO TABLE %s.%s LINES TERMINATED BY '\r\n' """ % \
            (tmpFile.replace('\\','/'), mode, dbName, tblName)

    logging.debug(query)
    rv = conn.execute(query)
    logging.info("Number of rows affected: %s" % len(df))

    # if (unlinkFile.lower() == 'yes') | (unlinkFile.lower() == 'y'):
    if unlinkFile:
        os.unlink(tmpFile)
        logging.info("Deleting temporary file: {}".format(tmpFile))
    logging.info("DONE")

    return rv
