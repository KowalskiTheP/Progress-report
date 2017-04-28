import readConf
import loadData

import time


config = readConf.readINI("../Data/config.conf")

if __name__=='__main__':
  
  global_start_time = time.time()
  
  loadData_start_time = time.time()
  print '> Loading data... '
  
  dataframe = loadData.load_fromCSV(config['csvfile'], 
			   ',', 
			   ';', 
			   int(config['header']),
			   config['datecolumn'])

  dataSet = loadData.getDataSet(dataframe, config['columns'])
  x_full, y_full = loadData.shiftData(dataSet, config['y_column'], int(config['look_back']))
  x_win, y_win = loadData.get_windows(x_full,y_full,int(config['winlength']))
  
  print '> Data loaded! This took: ', time.time() - loadData_start_time, 'seconds'
  
  


