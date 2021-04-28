from __future__ import print_function
# --------------------------------------------------------------------------------------------------
# backtest_v3,py
#
#
# --------------------------------------------------------------------------------------------------
# created by Joseph Loss on 4/5/2021

use_GUI = True

# --------------------------------------------------------------------------------------------------
# input parameters

targetVar = 'rtnTodayToTomorrow',
nTest = 100,
nTrain = 489,
winsorize = True,
stationary = True,
preprocessing = True,

model_parameters = {
        'rf_criterion':    'mse',               # mse, mae
        'max_features':    'auto',              # auto, sqrt, log2, None
        'n_estimators':    1000,
        'min_samples_leaf':100,
        'random_state':    42,
        'n_jobs':          -1,
}

ticker = 'SPY'
LOG_LEVEL = 'INFO'

# --------------------------------------------------------------------------------------------------
# Module imports

import os, logging, time
from lxml import _elementpath as _dummy
import gzip
import lxml
import gooey
import glob, sys, json, datetime as dt
from gooey import Gooey, GooeyParser, local_resource_path, PrefixTokenizers
from collections import OrderedDict, defaultdict
import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime as dt
import warnings
warnings.simplefilter("ignore")

from fastai.tabular.all import *
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from scipy.stats.mstats import winsorize
from statsmodels.tsa.stattools import adfuller

# custom imports
sys.path.append('./walk-forward-pred')
from fnLibrary import setPandas, fnUploadSQL, setOutputFilePath, setLogging, fnOdbcConnect
LOG_FILE_NAME = os.path.basename(__file__)


# --------------------------------------------------------------------------------------------------
# GUI

@Gooey(program_name = "Walk-Forward Random Forest Model", menu = [{
        'name': 'Menu',
        'items':[
                {
                        'type':     'AboutDialog',
                        'menuTitle':'About',
                        'name':     'Walk-Forward Random Forest Model',
                        'version':  '1.0',
                        'developer':'Joseph Loss',
                        'copyright':'Joseph Loss, Adam Schran',
                },
        ]
}],
       advanced = True,
       show_restart_button = False,
       clear_before_run = True,
       poll_external_updates = False,
       tabbed_groups = True,
       navigation = 'Tabbed',
       default_size = [
               610,
               750
       ],
   )
def parse_args():

    # parser = GooeyParser(description = "Calculate MWR for selected accounts.")
    parser = GooeyParser()
    # subs = parser.add_subparsers(help = 'commands', dest = 'subparser_name')
    # subMain = subs.add_parser('Main')

    # --------------------------------------------------------------------------------------------------
    # main group

    main = parser.add_argument_group("Main", gooey_options = { 'show_border':True, 'columns':2, })

    main.add_argument("--FileChooser", help="Please enter the data path (.csv)", widget="FileChooser", dest="df")
    main.add_argument('--targetVar', action='store', widget='TextField', help = 'target Y-variable')
    main.add_argument('--nTest',
                            action = 'store',
                            widget = 'TextField',
                            help = 'Rolling Test size',
                            default = 100,
                            gooey_options = { 'visible':True }
                            )
    main.add_argument('--nTrain',
                            action = 'store',
                            widget = 'TextField',
                            help = 'Rolling Train size',
                            # default = 100,
                            gooey_options = { 'visible':True }
                            )


    # --------------------------------------------------------------------------------------------------
    # main group

    boolParams = main.add_argument_group('Walk-Forward options', gooey_options = {
            'full_width':False, 'show_border':True, 'columns':2
            # "initial_selection": 0,
    })
    boolParams.add_argument('--winsorize',
                           metavar = 'Winsorize Outliers',
                           action = "store_true",
                           widget = 'BlockCheckbox',
                           # help='Historical Simulation',
                           default = True,
                           gooey_options = {
                                   # 'block_label':'Historical Simulation',
                                   'checkbox_label':'Include'
                           })

    boolParams.add_argument('--stationary',
                           metavar = 'Stationarity Test',
                           action = "store_true",
                           widget = 'BlockCheckbox',
                           # help='Historical Simulation',
                           default = True,
                           gooey_options = {
                                   # 'block_label':'Historical Simulation',
                                   'checkbox_label':'Include'
                           })

    boolParams.add_argument('--preprocess',
                           metavar = 'Data Preprocessing / Standardization',
                           action = "store_true",
                           widget = 'BlockCheckbox',
                           # help='Historical Simulation',
                           default = True,
                           gooey_options = {
                                   # 'block_label':'Historical Simulation',
                                   'checkbox_label':'Include'
                           })

    modelParams = parser.add_argument_group("Model Parameters", gooey_options = { 'show_border':True, 'columns':2, })

    modelParams.add_argument('--rf_criterion',
                            action = 'store',
                            widget = 'TextField',
                            help = 'Criterion for RF Model',
                            default = 'mse',
                            gooey_options = { 'visible':True }
                            )
    modelParams.add_argument('--max_features',
                            action = 'store',
                            widget = 'TextField',
                            help = 'Maximum number of features',
                            default = 'auto',
                            gooey_options = { 'visible':True }
                            )
    modelParams.add_argument('--n_estimators',
                            action = 'store',
                            widget = 'TextField',
                            help = 'Number of RF Estimators',
                            default = 1000,
                            gooey_options = { 'visible':True }
                            )
    modelParams.add_argument('--min_samples_leaf',
                            action = 'store',
                            widget = 'TextField',
                            help = 'Minimum samples per leaf',
                            default = 100,
                            gooey_options = { 'visible':True }
                            )
    modelParams.add_argument('--random_state',
                            action = 'store',
                            widget = 'TextField',
                            help = 'Random Generator',
                            default = 42,
                            gooey_options = { 'visible':True }
                            )
    modelParams.add_argument('--n_jobs',
                            action = 'store',
                            widget = 'TextField',
                            help = 'Number of Jobs',
                            default = -1,
                            gooey_options = { 'visible':True }
                            )

    args = parser.parse_args()
    return dict(args._get_kwargs())


# --------------------------------------------------------------------------------------------------
# walk forward module

def fnWalkForward(df=None, targetVar=None, nTest=None, nTrain=100, winsorize=True, stationary=True, preprocessing=True, **modelParams):

    # set numpy float format
    floatFormatter = "{:,.6f}".format
    np.set_printoptions(formatter = {'float_kind':floatFormatter})

    # set nTest (rolling every x days)
    nTest = nTest + 1

    dpred = { }
    dfLast = { }

    for i in range(0, len(df) - nTrain - nTest, 1):
        dpred, dfLast = predict(
                df[i:nTrain + nTest + i],
                nTrain,
                nTest,
                targetVar,
                winsorize,
                stationary,
                preprocessing,
                **modelParams)

    dpred = dpred.iloc[:-1]

    return dpred, dfLast


# --------------------------------------------------------------------------------------------------
# winsorize data method

def winsorizeData(s):
    return winsorize(s, limits = [0.005, 0.005])


# --------------------------------------------------------------------------------------------------
# adf testing

def adf_test(timeSeries):
    dfADF = adfuller(timeSeries, autolag = 'AIC')
    output = pd.Series(dfADF[0:4], index = ['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    for key, value in dfADF[4].items():
        output['Critical Value (%s)' % key] = value
    logging.debug('ADF Testing: %s \n%s\n' % (timeSeries.name, output))

    return output


# --------------------------------------------------------------------------------------------------
# test stationarity

def stationarity(result):
    plist = { }
    for col in result:
        if adf_test(result[col])['p-value'] < 0.05:
            st = True
        else:
            st = False
        plist[col] = st

    return plist


# --------------------------------------------------------------------------------------------------
# predict

def predict(df, nTrain, nTest, targetVar, winsorize, stationary, preprocessing, **modelParams):

    if modelParams['random_state'] is None:
        pass
    else:
        np.random.seed(seed = modelParams['random_state'])

    X_train = df[0:nTrain]
    X_test = df[nTrain:nTrain + nTest]

    y_train = X_train[targetVar]
    y_test = X_test[targetVar]

    # drop y variables from features
    X_train.drop(targetVar, axis = 1,)
    X_test.drop(targetVar, axis = 1)


    # --------------------------------------------------------------------------------------------------
    # winsorize / feature scaling

    if winsorize:
        X_train = X_train.apply(winsorizeData, axis = 0)
        maxTrain = X_train.max()
        minTrain = X_train.min()

        conditions = [(X_test.values < minTrain.values), (X_test.values > maxTrain.values)]
        choices = [minTrain, maxTrain]
        tmp = np.select(conditions, choices, default = X_test)

        X_test = pd.DataFrame._from_arrays(tmp.transpose(), columns = X_test.columns, index = X_test.index)

    else:
        X_train = X_train
        X_test = X_test


    # --------------------------------------------------------------------------------------------------
    # test for stationarity

    checkVals = X_train.loc[
        (X_train.values == -np.inf) |
        (X_train.values == np.inf) |
        (X_train.values == np.nan)
        ]

    if not checkVals.empty:
        logging.warning('-INF / +INF / NAN VALUES DETECTED. REPLACING WITH ZERO VALS')

    X_train.replace(
            (-np.inf, np.inf),
            np.nan,
            inplace = True
    )
    X_train.fillna(0,inplace=True)

    if stationary:
        stationarityResults = (stationarity(X_train))
        stationarityResults = pd.DataFrame(stationarityResults, index = [0])
        stationaryFactors = []

        for i in stationarityResults.columns:
            if stationarityResults[i][0] == 1:
                stationaryFactors.append(i)

        X_test = X_test[stationaryFactors].drop(targetVar,
                                                axis = 1)
        X_train = X_train[stationaryFactors].drop(targetVar,
                                                  axis = 1)
    else:
        X_test = X_test.drop(targetVar,
                             axis = 1)
        X_train = X_train.drop(targetVar,
                               axis = 1)


# --------------------------------------------------------------------------------------------------
    # Preprocess / Standardize data

    if preprocessing:

        sc_X = StandardScaler()
        X_train_std = sc_X.fit_transform(X_train)
        X_test_std = sc_X.fit_transform(X_test)

        y_train = np.array(y_train).reshape(-1, 1)
        y_test = np.array(y_test).reshape(-1, 1)

    else:
        X_train_std = X_train.copy()
        X_test_std = X_test.copy()


    # --------------------------------------------------------------------------------------------------
    # init random forest

    RFmodel = RandomForestRegressor(criterion = modelParams['rf_criterion'],
                                    max_features = modelParams['max_features'],
                                    n_estimators = modelParams['n_estimators'],
                                    # max_leaf_nodes = modelParams['max_leaf_nodes'],
                                    # oob_score = modelParams['oob_score'],
                                    # max_depth = modelParams['max_depth'],
                                    min_samples_leaf = modelParams['min_samples_leaf'],
                                    # min_samples_leaf = 100,
                                    random_state = modelParams['random_state'],
                                    n_jobs = modelParams['n_jobs'])

    RFmodel.fit(X_train_std, y_train)

    temp = pd.Series(RFmodel.feature_importances_, index = X_test.columns)
    # dfFeat[df.index[-1]] = temp

    # predict in-sample and out-of-sample
    y_train_pred = RFmodel.predict(X_train_std)
    y_test_pred = RFmodel.predict(X_test_std)


    # --------------------------------------------------------------------------------------------------
    # statistics

    print('\n\n----------------------------------------\n')
    logging.info("DATE: %s " % (pd.to_datetime(df.index[len(df) - 1]).date()))

    logging.info('MSE - Train: %.6f, Test: %.6f' % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred)))

    logging.info('R^2 - Train: %.4f, Test: %.4f' % (
            r2_score(y_train, y_train_pred),
            r2_score(y_test, y_test_pred)))

    logging.info('Explained Variance - Train: %.4f, Test: %.4f' % (
            explained_variance_score(y_train, y_train_pred),
            explained_variance_score(y_test, y_test_pred)))

    predictionsY = y_test_pred
    last_pred = predictionsY[-1]

    return predictionsY, last_pred


# --------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------
# run main

if __name__ == '__main__':

    warnings.simplefilter('ignore', category = FutureWarning)
    warnings.simplefilter(action = 'ignore', category = DeprecationWarning)
    warnings.simplefilter(action = 'ignore', category = PendingDeprecationWarning)
    warnings.filterwarnings('ignore')

    # custom pandas settings
    setPandas()
    setLogging(LOG_FILE_NAME = LOG_FILE_NAME, level = LOG_LEVEL)
    path = '..\\'

    # set numpy float format
    floatFormatter = "{:,.6f}".format
    np.set_printoptions(formatter = {'float_kind':floatFormatter})


    # --------------------------------------------------------------------------------------------------
    # --------------------------------------------------------------------------------------------------
    # todo:
    # TESTING

    df = pd.read_pickle('data.pkl')
    df.drop(columns = ['rtnTodayToTomorrowClassified'], inplace = True)

    # --------------------------------------------------------------------------------------------------


    if use_GUI:
        conf = parse_args()
        params

    else:
        model_parameters = model_parameters


    print("----- Running Tax Loss Harvesting Monte Carlo -----\n\n", flush = True)
    print('-------------------------', flush = True)
    print('INPUTS:', flush = True)
    print('email: {}'.format(conf.emailAddress), flush = True)
    print('ticker: {}'.format(conf.ticker_tk), flush = True)
    print('market value: {:,.2f}'.format(float(conf.marketValue)), flush = True)
    print('cost basis: {:.2f}%'.format(float(conf.costBasis) * 100), flush = True)
    if conf.historicalSim:
        print('include historical simulation: TRUE', flush = True)
    else:
        print('include historical simulation: FALSE', flush = True)
    print('benchmark: {}'.format(conf.bmarkTicker), flush = True)
    print('market return: {:.2f}%'.format(float(conf.marketReturn) * 100), flush = True)
    print('alpha: {:.2f}%'.format(float(conf.alpha) * 100), flush = True)
    print('historical skew: {}'.format(conf.histSkew), flush = True)
    print('nTrials: {:,.0f}'.format(float(conf.nTrials)), flush = True)
    print('option tenor: {:,.0f}'.format(float(conf.optTenor)), flush = True)
    print('nYears: {:,.0f}'.format(float(conf.nYears)), flush = True)
    print('state tax rate: {:.2f}%'.format(float(conf.stateTaxRate) * 100), flush = True)
    print('short term capital gains tax: {:.2f}%'.format(float(conf.stcg) * 100), flush = True)
    print('long term capital gains tax: {:.2f}%'.format(float(conf.ltcg) * 100), flush = True)
    print('-------------------------\n', flush = True)


    try:

        model_parameters = {
                'rf_criterion':    'mse',
                'max_features':    'auto',
                'n_estimators':    1000,
                'min_samples_leaf':100,
                'random_state':    42,
                'n_jobs':          -1,
                # 'max_leaf_nodes':2,
                # 'oob_score':True,
                # 'max_depth':10
        }

        dpred, dfLast = fnWalkForward(df,
                                      targetVar = 'rtnTodayToTomorrow',
                                      nTest = 100,
                                      nTrain = 489,
                                      winsorize = True,
                                      stationary = True,
                                      preprocessing = True,
                                      **model_parameters)

        logging.info("========== END PROGRAM ==========")

    except Exception as e:
        logging.error(str(e), exc_info=True)

    # CLOSE LOGGING
    for handler in logging.root.handlers:
        handler.close()
    logging.shutdown()
