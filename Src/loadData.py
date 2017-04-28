import numpy as np
import pandas as pd
from datetime import datetime

dax_df=pd.read_csv('../Data/dax_19700103_20170412.csv', sep=';', header=0)

# Some definitions and initialisation
trainData = np.zeros((len(dax_df), 3))
refdate = '01.01.1900'
date_format = "%d.%m.%Y"
b = datetime.strptime(refdate, date_format)

for i in range(len(dax_df)):
# The next 7 lines are needed to convert the date format in "%d.%m.%Y". 
# If the data already has this format nothing should happen.
  d=dax_df['Datum'][i].split('.')[0]
  m=dax_df['Datum'][i].split('.')[1]
  y=dax_df['Datum'][i].split('.')[2]
  if int(y)>17 and int(y)<1000:
    y=str(int(y)+1900)
  else:
    y=str(int(y)+2000)
  origDate = str(d+'.'+m+'.'+y)
  a = datetime.strptime(origDate, date_format)
  trainData[i,0] = int(str(a-b).split(' ')[0])
  
# The original data had commas as decimal signs. With sed all commas were replaced with points. 
# Now there are numbers like 1.000.54 which are catched by the following 'try' statemants.  
  try:
    trainData[i][1] = float(dax_df['Eroeffnung'][i])
  except:
    tmpEroeffnung = str(dax_df['Eroeffnung'][i]).split('.')
    trainData[i][1] = float(str(tmpEroeffnung[0]+tmpEroeffnung[1]+'.'+tmpEroeffnung[2]))   
  try:
    trainData[i][2] = float(dax_df['Schluss'][i])
  except:
    tmpSchluss = str(dax_df['Schluss'][i]).split('.')
    trainData[i][2] = float(str(tmpSchluss[0]+tmpSchluss[1]+'.'+tmpSchluss[2]))


print trainData
