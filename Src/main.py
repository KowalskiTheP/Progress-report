import readConf
import model

config = readConf.readINI("../Data/config.conf")

print xrange(1,len(config['neuronsperlayer']))

model = model.build_model(config)
