import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

## loading data from a CSV file in a pandas DataFrame. 
## The diff between the values in dateColumn and the refdate gets also calculated and added.
## If no date is aviable, dataColumn has to be 'None'
def load_fromCSV(csvFile, decimal , seperator, header, dateColumn):
  df=pd.read_csv(csvFile, decimal=decimal ,sep=seperator, header=header)
  if dateColumn != 'None':
    ## Some definitions and initialisation
    refdate = '01.01.1900'
    date_format = "%d.%m.%Y"
    date_list = []
    b = datetime.strptime(refdate, date_format)
    ## Getting the diff between ref and data date
    for i in range(len(df)):
      a = datetime.strptime(df.loc[i,dateColumn], date_format)
      date_list.append(a-b)
    df['days'] = date_list   
  return df

## Creating a numpy array with all data from the selected columns 
## of the previosly initialised dataFrame
def getDataSet(dataframe, columns):
  dataSet = np.zeros((len(dataframe), len(columns)))
  for i in range(len(columns)):
    dataSet[:,i] = dataframe.iloc[:,columns[i]]
  return dataSet

## Shifting the data by look_back to create usefull x and y arrays
def shiftData(data, y_column, look_back):
  x = np.zeros((len(data)-look_back,data.shape[1]))
  y = np.zeros((len(data)-look_back,1))
  for i in range(len(data)-look_back):
    x[i] = data[i]
    y[i] = data[i+1,y_column]
  return x, y

## The single windows will be the samples for the model
def get_windows(x,y,winLength):
  x_train, y_train = [], []
  for i in range(len(x)-(winLength+1)):
    x_train.append(x[i:i+winLength])
    y_train.append(y[i+winLength-1])
  return np.array(x_train), np.array(y_train)

def normalise_data(x_data, y_data):
  scaler = MinMaxScaler(feature_range=(0,1))
  scaler.fit(x_data)
  x_scaled = scaler.transform(x_data)
  y_scaled = scaler.transform(y_data)
  return x_scaled, y_scaled
  

## Following lines were used to test the functions  

#dataframe = load_fromCSV('../Data/dax_19700105_20170428.csv', ',', ';', 0,'Datum')
#print dataframe
#dataSet = getDataSet(dataframe, [4])
#x_full, y_full = shiftData(dataSet, 1)
#x_win, y_win = get_windows(x_full,y_full,10)

#print 'x: ',x_win[-3:-1]
#print 'y: ',y_win[-3:-1]

