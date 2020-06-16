import pandas as pd
from scipy import stats

backtestExcel = 'C://Users//jloss//PyCharmProjects//SMA-HullTrading-Practicum//Source Code//week 10//tmp.xlsx'

df = pd.read_excel(backtestExcel, header=0)
df.columns = [col.lower().strip() for col in df.columns]
print(beta, alpha)
1.027547807352036 -3.753017427403902e-05
print(beta.round(3), alpha.round(3))
1.028 -0.0
print(beta.round(3), alpha.round(7))
1.028 -3.75e-05
print(beta.round(3), alpha.round(5))
1.028 -4e-05
print(beta, alpha)
1.027547807352036 -3.753017427403902e-05
print(beta.round(5), alpha.round(5))
1.02755 -4e-05
print(beta.round(5), alpha.round(10))
1.02755 -3.75302e-05
# SML is described as E[Ri] = rf + B* (E[R_mkt] - rf)
def SML(rf, Rm, beta):
    ExpectedRet = rf + beta * (Rm - rf)
    return ExpectedRet
SML(rf=.1/100, Rm=.07*sqrt(252), beta = beta)

# # basic string cleaning:
# def clean_message_types(df):
#     df.value = df.value.str.strip()
#     df.name = (df.name
#                .str.strip() # remove whitespace
#                .str.lower()
#                .str.replace(' ', '_')
#                .str.replace('-', '_')
#                .str.replace('/', '_'))
#     df.notes = df.notes.str.strip()
#     df['message_type'] = df.loc[df.name == 'message_type', 'value']
#     return df
#
# # read message types from xlsx and run string cleaning function
# message_types = clean_message_types(message_data)
#
# # extract message type codes/names to make results more readable
# message_labels = (message_types.loc[:, ['message_type', 'notes']]
#                   .dropna()
#                   .rename(columns={'notes': 'name'}))
# message_labels.name = (message_labels.name
#                        .str.lower()
#                        .str.replace('message', '')
#                        .str.replace('.', '')
#                        .str.strip().str.replace(' ', '_'))
# print(message_labels.head())
#
# # finalize msg specs: offset, length, and value type to be used by struct
# message_types.message_type = message_types.message_type.ffill()
# message_types = message_types[message_types.name != 'message_type']
# message_types.value = (message_types.value
#                        .str.lower()
#                        .str.replace(' ', '_')
#                        .str.replace('(', '')
#                        .str.replace(')', ''))
