import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as stats
from tabulate import tabulate

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
    
    print '> Build time : ', time.time() - start
    
    start = time.time()
    if params['optimiser'] == 'adam':
        opt = Adam(lr = float(params['learningrate']))
    model.compile(loss=params['loss'], optimizer=opt)
    print '> Compilation Time : ', time.time() - start
    return model




def predict_point_by_point(model, data):
    #Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    predicted = model.predict(data)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted




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
    axes.set_ylim([-0.1,1.1])
    plt.show()




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
    
    
    
