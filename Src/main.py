import readConf
import model
import loadData

import time
import numpy as np


config = readConf.readINI("../Data/config.conf")

  
global_start_time = time.time()
  
loadData_start_time = time.time()
print '> Loading data... '
  
dataframe = loadData.load_fromCSV(config['csvfile'], ',', ';', int(config['header']), config['datecolumn'])

dataSet = loadData.getDataSet(dataframe, config['columns'])
x_full, y_full = loadData.shiftData(dataSet, config['y_column'], int(config['look_back']))
x_win, y_win = loadData.get_windows(x_full,y_full,int(config['winlength']))

print np.shape(x_win)
print np.shape(y_win)
print x_win[0,:,0]

x_tmp, y_tmp = [], []
for i in range(len(x_win)):
  x_tmp_scaled, y_tmp_scaled = loadData.normalise_data(x_win[i,:,0], y_win[i])
  x_tmp.append(x_tmp_scaled)
  y_tmp.append(y_tmp_scaled)
x_win_scaled = np.array(x_tmp).reshape(len(x_tmp),int(config['winlength']),1)
y_win_scaled = np.array(y_tmp).reshape(len(x_tmp),1)

print x_win_scaled
print y_win_scaled
  
print '> Data loaded! This took: ', time.time() - loadData_start_time, 'seconds'


model = model.build_model(config)

model.fit(x_win_scaled, y_win_scaled, int(config['batchsize']), int(config['epochs']))
