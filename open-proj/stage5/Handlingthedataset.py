import pandas as pd
import numpy as np
dfcar = pd.read_excel('input_data/Dataset.xlsx', sheet_name='Carbon Footprint',
                                        skiprows=1, index_col = 0)

cartable = pd.melt(dfcar, id_vars=['Activity', 'Per'],
                          value_vars=dfcar.columns.values[2:],
                          var_name='Name of Resource Used',
                          value_name='Carbon Footprint of Resource per Unit') #choosing columns
cartabledropna = cartable.dropna(axis = 0)
cartablezero= cartable
cartablezero['Carbon Footprint of Resource per Unit'] = np.nan_to_num(cartablezero['Carbon Footprint of Resource per Unit'])
cartable.to_csv('data/output_data/CarbonfootprintresourcesReshaping.csv', index=False) #The cleaned data was saved as a csv file
