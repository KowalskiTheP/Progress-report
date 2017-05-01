import readConf
import model
import loadData

import os
import time
import numpy as np
import sys

#import matplotlib.pyplot as plt

config = readConf.readINI("../Data/config.conf")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(config['loglevel'])
  
global_start_time = time.time()
  
loadData_start_time = time.time()
print '> Loading data... '
  
dataframe = loadData.load_fromCSV(config['csvfile'], ',', ';', int(config['header']), config['datecolumn'])

if config['windoweddata'] == 'on':
  
  print '> Windowing data..be patient little Padawan'
#  x_winTrain, y_winTrain, x_winTest, y_winTest, scaler = loadData.make_windowed_data(dataframe, config)

#  x_winTrain, y_winTrain, x_winTest, y_winTest, refValue = loadData.make_windowed_data_rewied(dataframe, config)
  
 

  x_winTrain, y_winTrain, x_winTest, y_winTest, trainRef, testRef = loadData.make_windowed_data_normOnWin(dataframe,config)
  
  print 'Shape of "x_winTrain"',x_winTrain.shape
  
  print 'Length of "y_winTrain"',len(y_winTrain)
  print x_winTest[-3:-1], y_winTest[-3:-1]

else:
  print 'not implemented so far, exiting!'
  sys.exit()

print '> Data loaded! This took: ', time.time() - loadData_start_time, 'seconds'

# build the specified model
model1 = model.build_model(config)

# train the model
model1.fit(x_winTrain, y_winTrain, int(config['batchsize']), int(config['epochs']))

# simple predictions or eval metrics
y_winTest = y_winTest.flatten()
y_winTrain = y_winTrain.flatten()

if config['evalmetrics'] == 'on':
  predTest = model.eval_model(x_winTest, y_winTest, model1, config, 'test data')
  predTrain = model.eval_model(x_winTrain, y_winTrain, model1, config, 'train data')
    
else:
  predTest = model.predict_point_by_point(model1, x_winTest)
  predTrain = model.predict_point_by_point(model1, x_winTrain)
  print np.column_stack((pred, y_winTest))


if config['plotting'] == 'on':
  
  #predTest = scaler.inverse_transform(predTest)
  #y_winTest = scaler.inverse_transform(y_winTest)
  #y_winTrain = scaler.inverse_transform(y_winTrain)
  #predTrain = scaler.inverse_transform(predTrain)
  
  #predTest = loadData.denormalise_data_refValue(refValue,predTest)
  #y_winTest = loadData.denormalise_data_refValue(refValue,y_winTest)
  #y_winTrain = loadData.denormalise_data_refValue(refValue,y_winTrain)
  #predTrain = loadData.denormalise_data_refValue(refValue,predTrain)
  
  print len(predTest), len(y_winTest)
  
  for i in range(len(testRef)):
    predTest[i] = testRef[i]*predTest[i]
    y_winTest[i] = testRef[i]*y_winTest[i]
  for i in range(len(trainRef)):
    y_winTrain[i] = trainRef[i]*y_winTrain[i]
    predTrain[i] = trainRef[i]*predTrain[i]
  
  model.plot_data(y_winTrain, predTrain)
  model.plot_data(y_winTest, predTest)
  model.plot_data(y_winTest[-10:-1], predTest[-10:-1])


