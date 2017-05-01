import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import datetime
import sys

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
def getDataSet(dataframe, columns, trainTestSplit):
  dataSet = np.zeros((len(dataframe), len(columns)))
  for i in range(len(columns)):
    dataSet[:,i] = dataframe.iloc[:,columns[i]]
  
  ############### maybe dont split train and test set randomly??? sounds crazy with a time series?
  #trainSet, testSet = train_test_split(dataSet, test_size = trainTestSplit)
  
  # first try last X percent of the dataset are the test set. PLEASE REVIEW!!!!
  # In principle it works now...but the data partioning has to be refined...COME ON RANDOM DATA GENERATOR!
  lastXpercent = int(np.floor(len(dataframe)*trainTestSplit))
  firstXpercent = len(dataframe) - lastXpercent
  
  trainSet = dataSet[0:firstXpercent]
  testSet = dataSet[firstXpercent:firstXpercent+lastXpercent]

  return trainSet, testSet


def split_data(dataSet, trainTestSplit):
  lastXpercent = int(np.floor(len(dataSet)*trainTestSplit))
  firstXpercent = len(dataSet) - lastXpercent
  trainSet = dataSet[0:firstXpercent]
  testSet = dataSet[firstXpercent:firstXpercent+lastXpercent]
  return trainSet, testSet


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
  return x_scaled, y_scaled, scaler


#def make_windowed_data(dataframe, config):
  #''' makes the windowed dataset from a appropriate dataframe'''
  
  #dataSetTrain, dataSetTest = getDataSet(dataframe, config['columns'], float(config['traintestsplit']))
  #dataSetTrain, dataSetTest, scaler = normalise_data(dataSetTrain, dataSetTest)
  ##print dataSetTrain
  ##print dataSetTest
  #x_fullTrain, y_fullTrain = shiftData(dataSetTrain, config['y_column'], int(config['look_back']))
  #x_winTrain, y_winTrain = get_windows(x_fullTrain,y_fullTrain,int(config['winlength']))
  #x_fullTest, y_fullTest = shiftData(dataSetTest, config['y_column'], int(config['look_back']))
  #x_winTest, y_winTest = get_windows(x_fullTest,y_fullTest,int(config['winlength']))
  
  #return x_winTrain, y_winTrain, x_winTest, y_winTest, scaler


def getDataSet_noSplit(dataframe, columns):
  dataSet = np.zeros((len(dataframe), len(columns)))
  for i in range(len(columns)):
    dataSet[:,i] = dataframe.iloc[:,columns[i]]
  return dataSet

def normalise_data_refValue(refValue,data):
  normData = []
  for i in range(len(data)):
    normData.append( data[i]/refValue - 0.0 )
  return np.array(normData)
  
def denormalise_data_refValue(refValue,normData):
  denormData = []
  for i in range(len(normData)):
    denormData.append( refValue * ( normData[i] + 0.0 ) )
  return np.array(denormData)

def get_windows_andShift(x,winLength,look_back):
  x_train, y_train = [], []
  for i in range(len(x)-(winLength+1)):
    x_train.append(x[i:i+winLength])
    y_train.append(x[i+winLength+look_back-1])
  return np.array(x_train), np.array(y_train)

def make_windowed_data_rewied(dataframe, config):
  refValue = 12000.0
  dataSet_Full = getDataSet_noSplit(dataframe, config['columns'])
#  dataSet_Full_norm =normalise_data_refValue(dataSet_Full[0,-1],dataSet_Full)
  dataSet_Full_norm = normalise_data_refValue(refValue,dataSet_Full)
  dataSetTrain_norm, dataSetTest_norm = split_data(dataSet_Full_norm, float(config['traintestsplit']))
  x_winTrain_norm, y_winTrain_norm = get_windows_andShift(dataSetTrain_norm,
                                                          int(config['winlength']),
                                                          int(config['look_back']))
  x_winTest_norm, y_winTest_norm = get_windows_andShift(dataSetTest_norm,
                                                          int(config['winlength']),
                                                          int(config['look_back']))
  return  x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm, refValue
  
  


## Following lines were used to test the functions  

#dataframe = load_fromCSV('../Data/dax_19700105_20170428.csv', ',', ';', 0,'Datum')
#print dataframe
#dataSet = getDataSet(dataframe, [4])
#x_full, y_full = shiftData(dataSet, 1)
#x_win, y_win = get_windows(x_full,y_full,10)

#print 'x: ',x_win[-3:-1]
#print 'y: ',y_win[-3:-1]

