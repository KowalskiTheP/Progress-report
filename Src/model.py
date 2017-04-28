import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
import numpy as np

def build_model(params):
    start = time.time()
    print "building model"
    
    # build sequential model
    model = Sequential()

    # first layer is special, gets build by hand
    print 'layer 0: ',params['neuronsperlayer'][0]
    model.add(LSTM(
      int(params['neuronsperlayer'][0]),
      input_shape = (None, int(params['inputdim'])),
      activation = str(params['activationperlayer'][0]),
      return_sequences=True,
      dropout = float(params['dropout'][0]),
      recurrent_activation = str(params['recurrentactivation'][0])
      )
    )

    # all interims layer get done by this for loop
    for i in xrange(1,len(params['neuronsperlayer'])-1):
      print 'layer ', i, ':', params['neuronsperlayer'][i]
      model.add(LSTM(
        int(params['neuronsperlayer'][i]),
        activation = str(params['activationperlayer'][i]),
        return_sequences=True,
        dropout = float(params['dropout'][i]),
        recurrent_activation = str(params['recurrentactivation'][i])
        )
      )
    
    #last LSTM layer is special because return_sequences=False
    print 'last LSTM layer: ',params['neuronsperlayer'][-1]
    model.add(LSTM(
      int(params['neuronsperlayer'][-1]),
      activation = str(params['activationperlayer'][-1]),
      return_sequences=False,
      dropout = float(params['dropout'][-1]),
      recurrent_activation = str(params['recurrentactivation'][-1])
      )
    )
    
    #last layer is dense
    print 'last layer (dense): ',params['outputdim']    
    model.add(Dense(
        units=int(params['outputdim']),
        activation = 'linear'
        )
    )
    
    print("> Build time : ", time.time() - start)
    
    start = time.time()
    if params['optimiser'] == 'adam':
        opt = Adam(lr = float(params['learningrate']))
    model.compile(loss=params['loss'], optimizer=opt)
    print("> Compilation Time : ", time.time() - start)
    return model

def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted

