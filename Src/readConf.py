from ConfigParser import SafeConfigParser
import sys

import pprint

def readINI(filename):
  """reads the ini file and checks if files and prerequisites are met"""
  #actual parsing
  confParser = SafeConfigParser()
  confParser.read(filename)

  configfileArgs={}

  for section_name in confParser.sections():
    for name, value in confParser.items(section_name):
      configfileArgs[name] = confParser.get(section_name, name)
  
  #print configfileArgs
  #print confParser.sections()
  for name, value in confParser.items('network'):
    if ',' in configfileArgs[name]:
      configfileArgs[name] = value.split(',')
  #print configfileArgs
  
  if isinstance(configfileArgs['neuronsperlayer'], list) == isinstance(configfileArgs['activationperlayer'], list):
    if len(configfileArgs['neuronsperlayer']) != len(configfileArgs['activationperlayer']):
      print 'number of activation functions has to be: 1 or equal to the length of neuronsPerLayer'
      sys.exit()
  else:
    temp_list1 = []
    for item in configfileArgs['neuronsperlayer']:
      temp_list1.append(configfileArgs['activationperlayer'])
    configfileArgs['activationperlayer'] = temp_list1

  if isinstance(configfileArgs['neuronsperlayer'], list) == isinstance(configfileArgs['recurrentactivation'], list):
    if len(configfileArgs['neuronsperlayer']) != len(configfileArgs['recurrentactivation']):
      print 'number of recurrent activation functions has to be: 1 or equal to the length of neuronsPerLayer'
      sys.exit()
  else:
    temp_list1 = []
    for item in configfileArgs['neuronsperlayer']:
      temp_list1.append(configfileArgs['recurrentactivation'])
    configfileArgs['recurrentactivation'] = temp_list1    
  
  if isinstance(configfileArgs['neuronsperlayer'], list) == isinstance(configfileArgs['dropout'], list):
    if len(configfileArgs['neuronsperlayer']) != len(configfileArgs['dropout']):
      print 'number of dropout has to be: 1 or equal to the length of neuronsPerLayer'
      sys.exit()
  else:
    temp_list1 = []
    for item in configfileArgs['neuronsperlayer']:
      temp_list1.append(configfileArgs['dropout'])
    configfileArgs['dropout'] = temp_list1
  
  pp = pprint.PrettyPrinter(indent=2)
  pp.pprint(configfileArgs)

      

 
  #for name, value in confParser.items('tuningParams'):
    #if configfileArgs[name] is not None:
      #if name == 'learningratetune':
        #valuesNew = [float(item) for item in configfileArgs[name]]
        #param['lr'] = valuesNew
      #elif name == 'batchsizetune':
        #valuesNew = [int(item) for item in configfileArgs[name]]
        #param['batch_size'] = valuesNew
      #elif name == 'nlayertune':
        #valuesNew = [int(item) for item in configfileArgs[name]]
        #param['number_of_layers'] = valuesNew
      #elif name == 'actlayertune':
        #valuesNew = [str(item) for item in configfileArgs[name]]
        #param['activation_layer'] = valuesNew
      #elif name == 'nhidunitsplayertune':
        #valuesNew = [int(item) for item in configfileArgs[name]]
        #param['n_hid_units_p_layer'] = valuesNew
      #elif name == 'l2regtune':
        #valuesNew = [float(item) for item in configfileArgs[name]]
        #param['Weight_reg'] = valuesNew
      #elif name == 'biastune':
        #valuesNew = [str(item) for item in configfileArgs[name]]
        #param['biasVar'] = valuesNew
      #elif name == 'dropouttune':
        #valuesNew = [float(item) for item in configfileArgs[name]]
        #param['dropout_hidden'] = valuesNew
      #elif name == 'batchnormtune':
        #valuesNew = [str(item) for item in configfileArgs[name]]
        #param['batch_norm'] = valuesNew
      #elif name == 'optimisertune':
        #valuesNew = [str(item) for item in configfileArgs[name]]
        #param['optimizerPar'] = valuesNew
      #elif name == 'losstune':
        #valuesNew = [str(item) for item in configfileArgs[name]]
        #param['lossV'] = valuesNew

  return configfileArgs
