import pandas as pd
import numpy as np
footprint = pd.read_excel('input_data/Dataset.xlsx', sheet_name='Individuals')
tableprint = pd.melt(footprint, id_vars=['Indnum', 'Group', 'Activity', 'Units', 'Consumption', 'Quality_of_Life_Importance__1_10'],
                                             value_vars=footprint.columns.values[6:], var_name='Name of Resource Used', value_name='Amount of Resource Used per Unit')
catable = tableprint.dropna(axis = 0)

catable.to_csv(r'data/output_data/Carbonfootprint_Reshaping.csv', index=False)#The clean file was saved as a csv file for Individuals
