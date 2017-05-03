import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.layers.normalization import BatchNormalization
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
from tabulate import tabulate

def build_model(params):
  '''builds model that is specified in params'''
  start = time.time()
  if int(params['verbosity']) < 2:
    print "building model"
  
  # build sequential model
  model = Sequential()

  # first layer is special, gets build by hand
  if int(params['verbosity']) < 2:
    print 'layer 0: ',params['neuronsperlayer'][0]
  model.add(LSTM(
    int(params['neuronsperlayer'][0]),
    input_shape = (None, int(params['inputdim'])),
    activation = str(params['activationperlayer'][0]),
    return_sequences=True,
    recurrent_activation = str(params['recurrentactivation'][0])
    )
  )
  if str(params['batchnorm']) == 'on':
    model.add(BatchNormalization())
  
  model.add(Dropout(float(params['dropout'][0])))

  # all interims layer get done by this for loop
  for i in xrange(1,len(params['neuronsperlayer'])-1):
    if int(params['verbosity']) < 2:
      print 'layer ', i, ':', params['neuronsperlayer'][i]
    
    model.add(LSTM(
      int(params['neuronsperlayer'][i]),
      activation = str(params['activationperlayer'][i]),
      return_sequences=True,
      recurrent_activation = str(params['recurrentactivation'][i])
      )
    )
    if str(params['batchnorm']) == 'on':
      model.add(BatchNormalization())

    model.add(Dropout(float(params['dropout'][i])))
  
  #last LSTM layer is special because return_sequences=False
  if int(params['verbosity']) < 2:
    print 'last LSTM layer: ',params['neuronsperlayer'][-1]
  model.add(LSTM(
    int(params['neuronsperlayer'][-1]),
    activation = str(params['activationperlayer'][-1]),
    return_sequences=False,
    recurrent_activation = str(params['recurrentactivation'][-1])
    )
  )
  if str(params['batchnorm']) == 'on':
    model.add(BatchNormalization())
  model.add(Dropout(float(params['dropout'][-1])))
  
  #last layer is dense
  if int(params['verbosity']) < 2:
    print 'last layer (dense): ',params['outputdim']    
  model.add(Dense(
      units=int(params['outputdim']),
      activation = 'linear'
      )
  )
  
  if int(params['verbosity']) < 2:
    print '> Build time : ', time.time() - start
  
  start = time.time()
  if params['optimiser'] == 'adam':
      opt = Adam(lr = float(params['learningrate']),
                 decay=float(params['decay']),
                 )
  model.compile(loss=params['loss'], optimizer=opt)
  
  if int(params['verbosity']) < 2:
    print '> Compilation Time : ', time.time() - start
  return model

###############################################

def predict_point_by_point(model, data):
  #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
  predicted = model.predict(data)
  predicted = np.reshape(predicted, (predicted.size,))
  return predicted

###############################################

def plot_data(true_data, pred_data, title='Your data'):
  '''makes simple plots of the evaluated data, nothing fancy'''
  
  #plt.ion()
  
  plt.title(title)
  plt.plot(true_data, ls='--', linewidth=2, color='tomato')
  plt.plot(pred_data, linewidth=2, color='indigo')
  tomato_patch = mpatches.Patch(color='tomato', label='true data')
  indigo_patch = mpatches.Patch(color='indigo', label='pred. data')
  plt.legend(handles=[tomato_patch,indigo_patch])
  axes = plt.gca()
  plt.autoscale(enable=True, axis='y')
  plt.show()

###############################################

def eval_model(test_x, test_y, trainedModel, config, tableHeader):
  '''calculate some core metrics for model evaluation'''
  
  score = trainedModel.evaluate(test_x, test_y, batch_size=int(config['batchsize']))
  pred = predict_point_by_point(trainedModel, test_x)
  rp, rp_P = stats.pearsonr(pred,test_y)
  rs, rs_P = stats.spearmanr(pred,test_y)
  sd = np.std(pred-test_y)
  print '------', tableHeader, '------'
  print tabulate({"metric": ['test loss', 'Rp', 'Rs', 'SD'],"model": [score, rp, rs, sd]}, headers="keys", tablefmt="orgtbl")
  np.savetxt(config['predictionfile'], np.column_stack((pred, test_y)), delimiter=' ')
  
  return pred
    
###############################################

def get_random_hyperparameterset(config):
  '''draws a random hyperparameter set when called'''
  
  params = {}
  params['nlayer_tune'] = int(config['nlayer_tune'][np.random.random_integers(0,len(config['nlayer_tune'])-1)])
  params['actlayer_tune'] = str(config['actlayer_tune'][np.random.random_integers(0,len(config['actlayer_tune'])-1)])
  params['nhiduplayer_tune'] = int(config['nhiduplayer_tune'][np.random.random_integers(0,len(config['nhiduplayer_tune'])-1)])
  params['dropout_tune'] = float(config['dropout_tune'][np.random.random_integers(0,len(config['dropout_tune'])-1)])
  
  temp = []
  temp1 = []
  temp2 = []
  for i in xrange(0,params['nlayer_tune']):
    
    temp.append(params['actlayer_tune'])
    temp1.append(params['nhiduplayer_tune'])
    temp2.append(params['dropout_tune'])
  
  config['neuronsperlayer'] = temp1
  config['activationperlayer'] = temp
  config['dropout'] = temp2
  config['learningrate'] = float(config['lr_tune'][np.random.random_integers(0,len(config['lr_tune'])-1)])
  config['batchsize'] = int(config['batchsize_tune'][np.random.random_integers(0,len(config['batchsize_tune'])-1)])
  config['batchnorm'] = str(config['batchnorm_tune'][np.random.random_integers(0,len(config['batchnorm_tune'])-1)])
  
  #print config['neuronsperlayer']
  #print config['activationperlayer']
  #print config['dropout']
  #print config['learningrate']
  #print config['batchsize']
  #print config['batchnorm']
  
  return config

###############################################

def run_nn(epochs, temp_config, X_train, Y_train):
  '''builds and the runs the specified model, after that it returns the last loss'''
  print temp_config['neuronsperlayer']
  model = build_model(temp_config)
  
  hist = model.fit(X_train, Y_train, epochs=epochs, batch_size=int(temp_config['batchsize']), verbose=0)
  
  last_loss = hist.history['loss'][-1]
  
  return last_loss

###############################################

def write_params(params, filename):
  '''writes the dictionary defined in params to a file'''
  
  with open(filename, 'w') as f:
    for key, value in config.items():
      f.write('%s: %s\n' % (key, value))
      
###############################################

def hypertune(X_train, Y_train, config):
  '''hyperband algorithm adapted from https://people.eecs.berkeley.edu/~kjamieson/hyperband.html'''
  
  start = time.time()
  print '> hyperparameter tuning through the hyperband algorithm will be done'
  
  max_iter = 2000  # maximum iterations/epochs per configuration
  eta = 3 # defines downsampling rate (default=3)
  logeta = lambda x: np.log(x)/np.log(eta)
  s_max = int(logeta(max_iter))  # number of unique executions of Successive Halving (minus one)
  B = (s_max+1)*max_iter  # total number of iterations (without reuse) per execution of Succesive Halving (n,r)

  #### Begin Finite Horizon Hyperband outerloop. Repeat indefinetely.
  for s in reversed(range(s_max+1)):
    print 'halving :', s
    n = int(np.ceil(B/max_iter/(s+1)*eta**s)) # initial number of configurations
    r = max_iter*eta**(-s) # initial number of iterations to run configurations for

    #### Begin Finite Horizon Successive Halving with (n,r)
    T = [ get_random_hyperparameterset(config) for i in range(n) ]
    for i in range(s+1):
      # Run each of the n_i configs for r_i iterations and keep best n_i/eta
      n_i = n*eta**(-i)
      r_i = r*eta**(i)
      val_losses = [ run_nn(int(r_i), t, X_train, Y_train) for t in T ]
      T = [ T[i] for i in np.argsort(val_losses)[0:int( n_i/eta )] ]
  
  
  filename = str(config['bestparams'])
  write_params(T[0], filename)
  
  print '> hyperparameter tuning took : ', time.time() - start
    
