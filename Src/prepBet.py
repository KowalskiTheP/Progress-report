import numpy as np 
import pandas as pd
from datetime import datetime, timedelta
import readConf
import numpy as np
import model

config = readConf.readINI("../Data/config.conf")
jsonFile = str(config['jsonfile'])
modelFile = str(config['modelfile'])
winLength = int(config['winlength'])
date2Pred = str(config['datetopred'])

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
  return dataSet

def selectRange(dateToPred, refdate, date_format, dataFormatedDate):
  a = datetime.strptime(dateToPred, date_format)
  b = datetime.strptime(refdate, date_format)
  c = str(a-b)
  c = [int(s) for s in c.split() if s.isdigit() ]
#  lowerBound = str(c - timedelta(days=np.ceil(winLength / 2) ))
#  lowerBound = [int(s) for s in lowerBound.split() if s.isdigit() ]
  for i in range(len(dataFormatedDate)):
    if dataFormatedDate.loc[i,'days'] == c:
#      dataFormatedDate.drop(i, axis=0,inplace=True)
      newdf = dataFormatedDate.loc[i-np.ceil(winLength / 2):i]
  return newdf

###############################################################################################

df_dax = pd.read_csv('../Data/dax_predWin.csv', decimal='.' ,sep=',', header=0)
df_nikkei = pd.read_csv('../Data/nikkei_predWin.csv', decimal='.' ,sep=',', header=0)
df_dowJones = pd.read_csv('../Data/dowJones_predWin.csv', decimal='.' ,sep=',', header=0)

df_dax.drop(['High','Low','Volume','Adj Close'], axis=1,inplace=True)
df_nikkei.drop(['High','Low','Volume','Adj Close'], axis=1,inplace=True)
df_dowJones.drop(['High','Low','Volume','Adj Close'], axis=1,inplace=True)

df_dax = convertDate(df_dax, 'Date', '%Y-%m-%d', '1985-01-01')
df_nikkei = convertDate(df_nikkei, 'Date', '%Y-%m-%d', '1985-01-01')
df_dowJones = convertDate(df_dowJones, 'Date', '%Y-%m-%d', '1985-01-01')

df_dax = selectRange(date2Pred, '1985-01-01','%Y-%m-%d' , df_dax)

df_combi = pd.merge(left=df_dax, right=df_nikkei, on='days')
df_combi = pd.merge(left=df_combi, right=df_dowJones, on='days')

df_dax = df_combi.loc[:,['days','Open_x','Close_x']]
df_nikkei = df_combi.loc[:,['days','Open_y','Close_y']]
df_dowJones = df_combi.loc[:,['days','Open','Close']]

array_dax = reoder(df_dax, [1,2])

array_nikkei = reoder(df_nikkei, [1,2])
array_dowJones = reoder(df_dowJones, [1,2])

for i in range(len(array_dax)):
  if array_dax[i,0]!=array_nikkei[i,0] or array_dax[i,0]!=array_dowJones[i,0] or array_dowJones[i,0]!=array_nikkei[i,0]:
    print 'Problem!!! Arrays are not in sync!'
    
df_dax = pd.DataFrame(data=array_dax, columns=['days','stock'])
df_nikkei = pd.DataFrame(data=array_nikkei, columns=['days','stock'])
df_dowJones = pd.DataFrame(data=array_dowJones, columns=['days','stock'])

#Shift nikkei (close=open)
df_nikkei = df_nikkei.drop(df_nikkei.index[[0]])
df_nikkei = df_nikkei.reset_index(drop=True)

df_combi = pd.concat([df_dax, df_nikkei['stock']], axis=1, join_axes=[df_dax.index])
df_combi = pd.concat([df_combi, df_dowJones['stock']], axis=1, join_axes=[df_combi.index])
df_combi.drop([winLength],axis=0,inplace=True)

print df_combi

toPred = df_combi.as_matrix()
multiplier = toPred[-1]
toPred = toPred / multiplier
toPred = toPred.reshape(1,winLength,4)

#fileList = ['epochs300_tanh_100100100', 'epochs300_tanh_757575', 'epochs500_tanh_100100100', 'epochs500_tanh_757575']
#sumPred = 0.
#for prefix in fileList:
  #jString = '../Data/'+prefix+'.json'
  #hString = '../Data/'+prefix+'.h5'
  #loaded_model = model.load_model(jString, hString)
  #betThis = loaded_model.predict(toPred)
  #betThis = betThis * multiplier[1]
  #sumPred = sumPred + betThis
  #print 'Bet this: ', betThis, ' !'
#print 'Mean: ', sumPred/len(fileList)

loaded_model = model.load_model(jsonFile, modelFile)
betThis = loaded_model.predict(toPred)
betThis = betThis * multiplier[1]
print 'Bet this: ', betThis, ' !'
