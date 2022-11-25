#!bin/zsh python 3.9


'''
-Staphylococcus aureus
-Outcome 1 : resistance, intermediate. 0: susceotible
-Target drug
    |--Ciprofloxacin
    |--Fusidic acid
    |--Oxacillin

- Drug name check
susc_data.columns
index = susc_data.columns.str.contains('(ciproflo)',regex=True,case=False)
susc_data.columns[index]
'''

import numpy as np
import pandas as pd
import os.path
import tracemalloc


ms_data = pd.read_csv("/Users/seibi/emory/cs534/CS534FinalMSpathogens/staph_ms.csv")
susc_data = pd.read_csv("/Users/seibi/emory/cs534/CS534FinalMSpathogens/staph_susceptibility.csv") 

ms_data.shape # 6994, 6001
susc_data.shape # 6994, 93


# check
susc_data['Fusidic acid'].value_counts()

# Ciprofloxacin
cipro_conditions = [
(susc_data["Ciprofloxacin"].isna() == True),
(susc_data["Ciprofloxacin"] == "S"),
(susc_data["Ciprofloxacin"] == 'R'),
(susc_data["Ciprofloxacin"] == 'I'),
(susc_data["Ciprofloxacin"] == 'R(1), S(1)'),
(susc_data["Ciprofloxacin"] == '-'),
(susc_data["Ciprofloxacin"] == 'I(1), S(1)'),
(susc_data["Ciprofloxacin"] == 'R(1), I(1)'),
(susc_data["Ciprofloxacin"] == 'R(1), I(1), S(1)')
]
values = [np.nan, 0,1,1,1,np.nan, 1,1,1]
susc_data['cipro_y'] = np.select(cipro_conditions, values)

# check
susc_data.groupby(["Oxacillin", 'oxa_y']).size()


# Fusidic acid
fusidic_conditions = [
(susc_data["Fusidic acid"].isna() == True),
(susc_data["Fusidic acid"] == "S"),
(susc_data["Fusidic acid"] == 'R'),
(susc_data["Fusidic acid"] == 'R(1), S(1)'),
(susc_data["Fusidic acid"] == '-')
]
values = [np.nan, 0,1,1,np.nan]
susc_data['fusidic_y'] = np.select(fusidic_conditions, values)


# Oxacillin
oxa_conditions = [
(susc_data["Oxacillin"].isna() == True),
(susc_data["Oxacillin"] == "S"),
(susc_data["Oxacillin"] == 'R'),
(susc_data["Oxacillin"] == '-'),
(susc_data["Oxacillin"] == 'R(1), S(1)')

]
values = [np.nan, 0,1,np.nan,1]
susc_data['oxa_y'] = np.select(oxa_conditions, values)


# complete dataset for each three drugs
def complete_case_(msdata, outcome):
    temp = pd.concat([ms_data, susc_data[outcome]], axis = 1)
    temp.dropna(axis = 0, how = 'any', inplace = True)
    temp.shape # 4517, 6002
    notkeep = ["Unnamed: 0",outcome]
    x = temp.drop(columns = notkeep)
    y = temp[outcome]
    return x, y

# create x, y file for each drugs
drugs = ['cipro','fusidic','oxa']
for i in drugs:
    outcome = i + '_y'
    x,y = complete_case_(ms_data, outcome)
    x_filename = i + '_x.csv'
    y_filename = i + '_y.csv'
    x.to_csv(x_filename, index=False)
    y.to_csv(y_filename, index=False)
