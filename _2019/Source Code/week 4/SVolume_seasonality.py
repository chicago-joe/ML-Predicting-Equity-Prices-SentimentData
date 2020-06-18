#
from pprint import pprint
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from patsy.highlevel import dmatrices, dmatrix



col_names = ['ticker', 'date', 'description', 'raw-s', 's-volume',
             's-dispersion',	'raw-s-delta','volume-delta', 'center-date',
             'center-time', 'center-time-zone']
# pd.read

SPY_2015_1min = pd.read_csv('C://Users//jloss//PyCharmProjects//SMA-HullTrading-Practicum//'
                            'Source Code//week 4//joey//Activity Feed Data//sma_history_tw_1m_activity_etf_201507_201512//'
                            'SPY.txt',sep='\s+',
                            # delimiter = '\s+',
                            header= 3,
                            # names=col_names,
                            # index_col = 1,
                            usecols=(6,8,9))


SPY_2016_1min = pd.read_csv('C://Users//jloss//PyCharmProjects//SMA-HullTrading-Practicum//'
                            'Source Code//week 4//joey//Activity Feed Data//sma_history_tw_1m_activity_etf_201701_201712//'
                            'SPY.txt',delimiter = '\t', header=3,names=col_names)
SPY_2017_1min = pd.read_csv('C://Users//jloss//PyCharmProjects//SMA-HullTrading-Practicum//'
                            'Source Code//week 4//joey//Activity Feed Data//sma_history_tw_1m_activity_etf_201601_201612//'
                            'SPY.txt',sep=',',delimiter = '\t', header=3,names=col_names)
print(SPY_2017_1min.columns)
print(SPY_2017_1min.head())

df15 = pd.DataFrame(SPY_2015_1min)
# df15 =
df16 = pd.DataFrame(SPY_2016_1min)
df17 = pd.DataFrame(SPY_2017_1min)
pprint(df17)


# def extractData(self, message):
#         trades = self.tradeMessage(message)
#         parsed_data=[trades[3],trades[7],trades[8],trades[6]]
#         hour=trades[3].split(':')[0]
#         min=trades[3].split(':')[1]
#         if self.flag is None:
#             self.flag = hour
#         elif self.flag != hour:
#             df = pd.DataFrame(self.temp, columns=['Time', 'Symbol', 'Price', 'Volume'])
#             if len(df)>0:
#                 result = self.calculate_VWAP(df)
#                 result.to_csv(os.path.join('.', 'output', str(self.flag) + '.csv'), sep=',', index=False)
#                 print(result)
#                 print("-------------------------------------\n")
#             if (int(self.flag) < 9):
#                 print("\nCurrent Time: ", self.flag,":00 AM (Eastern Time)")
#                 print('========== THE MARKET IS CURRENTLY CLOSED ==========\n\n')
#             if (int(self.flag) > 15):
#                 print("\nCurrent Time: ", self.flag,":00 PM (Eastern Time)")
#                 print('========== THE MARKET IS CURRENTLY CLOSED ==========\n\n')
#             self.temp = []
#             self.flag = None
#         else:
#             if self.TradeHour:
#                 self.temp.append(parsed_data)