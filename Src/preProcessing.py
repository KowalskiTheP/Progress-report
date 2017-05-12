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

def reoder(dataframe, columns):
  dataSet = np.zeros((len(dataframe)*2, 2))
  j=0
  for i in xrange(0,len(dataframe)*2,2):
    dataSet[i,0] = dataframe.iloc[j,0]
    dataSet[i,1] = dataframe.iloc[j,columns[0]]
    dataSet[i+1,0] = dataframe.iloc[j,0]
    dataSet[i+1,1] = dataframe.iloc[j,columns[1]]
    j=j+1
  print 'Data np.array:\n', dataSet, '\n'
  return dataSet

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

df_combi = pd.merge(left=df_dax, right=df_nikkei, on='days')
df_combi = pd.merge(left=df_combi, right=df_dowJones, on='days')

df_dax = df_combi.loc[:,['days','Eroeffnung','Schluss']]
df_nikkei = df_combi.loc[:,['days','Open_x','Close_x']]
df_dowJones = df_combi.loc[:,['days','Open_y','Close_y']]

array_dax = reoder(df_dax, [1,2])
array_nikkei = reoder(df_nikkei, [1,2])
array_dowJones = reoder(df_dowJones, [1,2])

print 'array_nikkei:\n', array_nikkei

for i in range(len(array_dax)):
  if array_dax[i,0]!=array_nikkei[i,0] or array_dax[i,0]!=array_dowJones[i,0] or array_dowJones[i,0]!=array_nikkei[i,0]:
    print 'Problem!!! Arrays are not in sync!'    


df_dax = pd.DataFrame(data=array_dax, columns=['days','stock'])
df_nikkei = pd.DataFrame(data=array_nikkei, columns=['days','stock'])
df_dowJones = pd.DataFrame(data=array_dowJones, columns=['days','stock'])

print 'df_nikkei:\n', df_nikkei

df_combi = pd.concat([df_dax, df_nikkei['stock']], axis=1, join_axes=[df_dax.index])
df_combi = pd.concat([df_combi, df_dowJones['stock']], axis=1, join_axes=[df_combi.index])

df_combi.to_csv('../Data/combi_dax_nikkei_dowJones.csv',index=False)
print df_combi



