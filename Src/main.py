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
  
  if config['normalise'] == '1':
    x_winTrain, y_winTrain, x_winTest, y_winTest, scaler = loadData.make_windowed_data(dataframe, config)  
  
  if config['normalise'] == '2':
    refValue = float(config['refvalue'])
    x_winTrain, y_winTrain, x_winTest, y_winTest = loadData.make_windowed_data_normOnFull(dataframe, config) 
    
  if config['normalise'] == '3':
    x_winTrain, y_winTrain, x_winTest, y_winTest, trainRef, testRef = loadData.make_windowed_data_normOnWin(dataframe,config)
    print trainRef[0]
    print trainRef[0,-1]

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
  

if config['normalise'] == '1':
  predTest = scaler.inverse_transform(predTest)
  y_winTest = scaler.inverse_transform(y_winTest)
  y_winTrain = scaler.inverse_transform(y_winTrain)
  predTrain = scaler.inverse_transform(predTrain)

if config['normalise'] == '2':
  refValue = float(config['refvalue'])
  predTest = loadData.denormalise_data_refValue(refValue,predTest)
  y_winTest = loadData.denormalise_data_refValue(refValue,y_winTest)
  y_winTrain = loadData.denormalise_data_refValue(refValue,y_winTrain)
  predTrain = loadData.denormalise_data_refValue(refValue,predTrain)
  
if config['normalise'] == '3':
  print trainRef
  for i in range(len(testRef)):
    predTest[i] = testRef[i,-1]*predTest[i]
    y_winTest[i] = testRef[i,-1]*y_winTest[i]
  for i in range(len(trainRef)):
    y_winTrain[i] = trainRef[i,-1]*y_winTrain[i]
    predTrain[i] = trainRef[i,-1]*predTrain[i]
  
diffTrain = np.sqrt((predTest - y_winTest)**2)
print 'Mean of pred.-true-diff:               ', np.mean(diffTrain)
print 'Standard deviation of pred.-true-diff: ', np.std(diffTrain)
      
if config['plotting'] == 'on':
  model.plot_data(y_winTrain, predTrain)
  model.plot_data(y_winTest, predTest)
  model.plot_data(y_winTest[-15:-1], predTest[-15:-1])
 



