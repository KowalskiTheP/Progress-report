import readConf
import model
import loadData

import time
import numpy as np

import matplotlib.pyplot as plt

config = readConf.readINI("../Data/config.conf")

  
global_start_time = time.time()
  
loadData_start_time = time.time()
print '> Loading data... '
  
dataframe = loadData.load_fromCSV(config['csvfile'], ',', ';', int(config['header']), config['datecolumn'])

dataSetTrain, dataSetTest = loadData.getDataSet(dataframe, config['columns'])
x_fullTrain, y_fullTrain = loadData.shiftData(dataSetTrain, config['y_column'], int(config['look_back']))
x_winTrain, y_winTrain = loadData.get_windows(x_fullTrain,y_fullTrain,int(config['winlength']))
x_fullTest, y_fullTest = loadData.shiftData(dataSetTest, config['y_column'], int(config['look_back']))
x_winTest, y_winTest = loadData.get_windows(x_fullTest,y_fullTest,int(config['winlength']))

#print np.shape(x_win)
#print np.shape(y_win)
#print x_win[0,:,0]

x_tmp, y_tmp = [], []
for i in range(len(x_winTrain)):
  x_tmp_scaled, y_tmp_scaled = loadData.normalise_data(x_winTrain[i,:,0], y_winTrain[i])
  x_tmp.append(x_tmp_scaled)
  y_tmp.append(y_tmp_scaled)
x_win_scaledTrain = np.array(x_tmp).reshape(len(x_tmp),int(config['winlength']),1)
y_win_scaledTrain = np.array(y_tmp).reshape(len(x_tmp),1)

x_tmp, y_tmp = [], []
for i in range(len(x_winTest)):
  x_tmp_scaled, y_tmp_scaled = loadData.normalise_data(x_winTest[i,:,0], y_winTest[i])
  x_tmp.append(x_tmp_scaled)
  y_tmp.append(y_tmp_scaled)
x_win_scaledTest = np.array(x_tmp).reshape(len(x_tmp),int(config['winlength']),1)
y_win_scaledTest = np.array(y_tmp).reshape(len(x_tmp),1)

print x_win_scaledTrain
print y_win_scaledTrain
  
print '> Data loaded! This took: ', time.time() - loadData_start_time, 'seconds'


model1 = model.build_model(config)

model1.fit(x_win_scaledTrain, y_win_scaledTrain, int(config['batchsize']), int(config['epochs']))


trainPredict = model.predict_point_by_point(model1, x_win_scaledTrain)
testPredict = model.predict_point_by_point(model1, x_win_scaledTest)



plt.plot(y_win_scaledTrain)
plt.plot(trainPredict)
axes = plt.gca()
axes.set_ylim([-0.1,1.1])
plt.show()

plt.plot(y_win_scaledTest)
plt.plot(testPredict)
axes = plt.gca()
axes.set_ylim([-0.1,1.1])
plt.show()
