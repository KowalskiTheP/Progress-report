import pandas as pd
import numpy as np
from datetime import datetime

def convertDate(df, dateColumn, date_format, refdate ):
  date_list = []
  b = datetime.strptime(refdate, date_format)
  for i in range(len(df)):
    a = datetime.strptime(df.loc[i,dateColumn], date_format)
    date_list.append(str(a-b))
    date_list[-1] = [int(s) for s in date_list[-1].split() if s.isdigit() ]
  date_list = np.array(date_list).flatten()
  df['days'] = date_list   
  return df

df_dax = pd.read_csv('../Data/dax_19700105_20170428.csv', decimal=',' ,sep=';', header=0)
df_dax = df_dax.drop(df_dax.index[range(0,3740)])
df_dax = df_dax.reset_index(drop=True)

df_nikkei = pd.read_csv('../Data/Nikkei_02051985_02052017.csv', decimal='.' ,sep=',', header=0)
df_dowJones = pd.read_csv('../Data/DowJones_02051985_02052017.csv', decimal='.' ,sep=',', header=0)


df_dax.drop(['Hoch','Tief','Performance','Volumen','Abstand Hoch/Tief'], axis=1,inplace=True)
df_nikkei.drop(['High','Low','Volume','Adj Close'], axis=1,inplace=True)
df_dowJones.drop(['High','Low','Volume','Adj Close'], axis=1,inplace=True)

df_dax = convertDate(df_dax, 'Datum', '%d.%m.%Y', '01.01.1985')
df_nikkei = convertDate(df_nikkei, 'Date', '%Y-%m-%d', '1985-01-01')
df_dowJones = convertDate(df_dowJones, 'Date', '%Y-%m-%d', '1985-01-01')

df_test = pd.merge(left=df_dax, right=df_nikkei, on='days')
df_test = pd.merge(left=df_test, right=df_dowJones, on='days')
print df_test
