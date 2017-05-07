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
    refdate = '01.01.1960'
    date_format = "%d.%m.%Y"
    date_list = []
    b = datetime.strptime(refdate, date_format)
    ## Getting the diff between ref and data date
    for i in range(len(df)):
      a = datetime.strptime(df.loc[i,dateColumn], date_format)
      date_list.append(str(a-b))
      date_list[-1] = [int(s) for s in date_list[-1].split() if s.isdigit() ]
    date_list = np.array(date_list).flatten()
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

###############################################

def normalise_data(x_data, y_data):
  scaler = MinMaxScaler(feature_range=(0,1))
  scaler.fit(x_data)
  x_scaled = scaler.transform(x_data)
  y_scaled = scaler.transform(y_data)
  return x_scaled, y_scaled, scaler

###############################################

def make_windowed_data(dataframe, config):
  ''' makes the windowed dataset from a appropriate dataframe'''
  
  dataSetTrain, dataSetTest = getDataSet(dataframe, config['columns'], float(config['traintestsplit']))
  dataSetTrain, dataSetTest, scaler = normalise_data(dataSetTrain, dataSetTest)
  x_fullTrain, y_fullTrain = shiftData(dataSetTrain, config['y_column'], int(config['look_back']))
  x_winTrain, y_winTrain = get_windows(x_fullTrain,y_fullTrain,int(config['winlength']))
  x_fullTest, y_fullTest = shiftData(dataSetTest, config['y_column'], int(config['look_back']))
  x_winTest, y_winTest = get_windows(x_fullTest,y_fullTest,int(config['winlength']))
  
  return x_winTrain, y_winTrain, x_winTest, y_winTest, scaler

###############################################

def getDataSet_noSplit(dataframe, columns):
  dataSet = np.zeros((len(dataframe), len(columns)))
  for i in range(len(columns)):
    dataSet[:,i] = dataframe.iloc[:,columns[i]]
  return dataSet


# loading data from csv and sequenzilce it ([[day_x1,startStock],[day_x1,endStock],[day_x2,startStock],[day_x2,endStock],...])

def getDataSet_noSplit_seq(dataframe, columns):
  dataSet = np.zeros((len(dataframe)*2, 2))
  j=0
  for i in xrange(0,len(dataframe)*2,2):
    dataSet[i,0] = dataframe.iloc[j,-1]
    dataSet[i,1] = dataframe.iloc[j,columns[0]]
    dataSet[i+1,0] = dataframe.iloc[j,-1]
    dataSet[i+1,1] = dataframe.iloc[j,columns[1]]
    j=j+1
  print 'Data np.array:\n', dataSet, '\n'
  return dataSet

###############################################

def normalise_data_refValue(refValue,data):
  normData = []
  for i in range(len(data)):
    normData.append( data[i]/refValue + 0.0 )
  return np.array(normData)

###############################################
  
def denormalise_data_refValue(refValue,normData):
  denormData = []
  for i in range(len(normData)):
    denormData.append( refValue * ( normData[i] + 0.0) )
  return np.array(denormData)

###############################################

def get_windows_andShift(x,winLength,look_back,outDim):
  x_train, y_train = [], []
  for i in range(len(x)-(winLength+outDim)):
    x_train.append(x[i:i+winLength])
    y_train.append(x[(i+winLength+look_back-1):(i+winLength+look_back+outDim-1),-1])
  print np.array(y_train)
  return np.array(x_train), np.reshape(np.array(y_train),(len(y_train),outDim))

###############################################

def get_windows_andShift_seq(x,winLength,look_back,outDim):
  x_train, y_train = [], []
  for i in xrange(0,len(x)-(winLength+outDim),2):
    x_train.append(x[i:i+winLength])
    y_train.append(x[(i+winLength+look_back-1):(i+winLength+look_back+outDim-1),-1])
  print np.array(y_train)
  return np.array(x_train), np.reshape(np.array(y_train),(len(y_train),outDim))

###############################################

def make_windowed_data_normOnFull(dataframe, config):
  refValue = float(config['refvalue'])
  winL = int(config['winlength'])
  lookB = int(config['look_back'])
  dataSet_Full = getDataSet_noSplit(dataframe, config['columns'])
#  dataSet_Full_norm =normalise_data_refValue(dataSet_Full[0,-1],dataSet_Full)
  dataSet_Full_norm = normalise_data_refValue(refValue,dataSet_Full)
  dataSetTrain_norm, dataSetTest_norm = split_data(dataSet_Full_norm, float(config['traintestsplit']))
  x_winTrain_norm, y_winTrain_norm = get_windows_andShift(dataSetTrain_norm, winL, lookB)
  x_winTest_norm, y_winTest_norm = get_windows_andShift(dataSetTest_norm, winL, lookB)
  return  x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm

###############################################

def make_windowed_data_normOnWin(dataframe, config):
  print dataframe
  refValue = float(config['refvalue'])
  winL = int(config['winlength'])
  lookB = int(config['look_back'])
  xDim = len(config['columns'])
  yDim = int(config['outputdim'])
  #dataSet_Full = getDataSet_noSplit(dataframe, config['columns'])
  dataSet_Full = getDataSet_noSplit_seq(dataframe, config['columns'])
  dataSetTrain, dataSetTest = split_data(dataSet_Full, float(config['traintestsplit']))
  #x_winTrain, y_winTrain = get_windows_andShift(dataSetTrain, winL, lookB,yDim)
  #x_winTest, y_winTest = get_windows_andShift(dataSetTest, winL, lookB,yDim)
  x_winTrain, y_winTrain = get_windows_andShift_seq(dataSetTrain, winL, lookB,yDim)
  x_winTest, y_winTest = get_windows_andShift_seq(dataSetTest, winL, lookB,yDim)
  
  print 'x_winTrain:\n', x_winTrain
  print 'y_winTrain:\n', y_winTrain
  
  x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm,trainRef,testRef = [],[],[],[],[],[]
  for i in range(len(y_winTrain)):
    x_winTrain_norm.append(normalise_data_refValue(x_winTrain[i,-1],x_winTrain[i]))
    y_winTrain_norm.append(normalise_data_refValue(x_winTrain[i,-1,-1],y_winTrain[i]))
    trainRef.append(x_winTrain[i,-1])
  for i in range(len(y_winTest)):
    x_winTest_norm.append(normalise_data_refValue(x_winTest[i,-1],x_winTest[i]))
    y_winTest_norm.append(normalise_data_refValue(x_winTest[i,-1,-1],y_winTest[i]))
    testRef.append(x_winTest[i,-1])
    
  x_winTrain_norm = np.reshape(np.array(x_winTrain_norm),(len(x_winTrain_norm),winL,xDim ))
  y_winTrain_norm = np.reshape(np.array(y_winTrain_norm),(len(y_winTrain_norm),yDim ))
  x_winTest_norm =  np.reshape(np.array(x_winTest_norm) ,(len(x_winTest_norm) ,winL,xDim ))
  y_winTest_norm =  np.reshape(np.array(y_winTest_norm) ,(len(y_winTest_norm) ,yDim ))
  
  return x_winTrain_norm, y_winTrain_norm, x_winTest_norm, y_winTest_norm, np.array(trainRef), np.array(testRef)

## Following lines were used to test the functions  

#dataframe = load_fromCSV('../Data/dax_19700105_20170428.csv', ',', ';', 0,'Datum')
#print dataframe
#dataSet = getDataSet(dataframe, [4])
#x_full, y_full = shiftData(dataSet, 1)
#x_win, y_win = get_windows(x_full,y_full,10)

#print 'x: ',x_win[-3:-1]
#print 'y: ',y_win[-3:-1]

