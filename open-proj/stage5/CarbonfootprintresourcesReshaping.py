import pandas as pd
import matplotlib.pyplot as matplt
import numpy as np
from sklearn import preprocessing #preprocessing of data
import seaborn as sb #seaborn package was imported

from sklearn import linear_model
s=pd.read_csv('output_data/CarbonfootprintresourcesReshaping.csv') #the file was read
s.columns=['X','Y','Z','P']
sb.set_context('notebook',font_scale = 1.5)
sb.set_style('whitegrid')
sb.lmplot('X','Y',data=s)
