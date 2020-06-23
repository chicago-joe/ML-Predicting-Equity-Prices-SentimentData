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

def setLogging(LOGGING_DIRECTORY = os.path.join('D:\\', 'logs', 'srAdvisors.v2', dt.today().strftime('%Y-%m-%d'), 'python'), LOG_FILE_NAME = os.path.basename(__file__) + '.log', level = 'INFO'):

    # reloads logging (useful for iPython only)
    reload(logging)

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


# --------------------------------------------------------------------------------------------------
# set up output filepath

def setOutputFile(OUTPUT_DIRECTORY = os.path.join('D:\\', 'tmp', 'advisorscodebase'), file = 'file'):

    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)

    path = os.path.join(OUTPUT_DIRECTORY, file)
    # print(path)
    return path


# --------------------------------------------------------------------------------------------------
#

# setOutputFilepath()


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
#
# def fnCreateOptKeys(df, key = 'secKey', retValue='optSeries'):
#     if retValue=='optSeries':
#         df['optKey'] = df[[key + '_tk',
#                            key + '_yr',
#                            key + '_mn',
#                            key + '_dy',
#                            key + '_xx',
#                            key + '_cp']].astype('str').agg(':'.join, axis = 1)
#         return df
#     elif retValue == 'tkSeries':
#         df['optKey'] = df[[key + '_yr',
#                            key + '_mn',
#                            key + '_dy',
#                            key + '_xx',
#                            key + '_cp']].astype('str').agg(':'.join, axis = 1)
#         return df
#     else:
#         print("Please choose a retValue: [tkSeries | optSeries] ")

