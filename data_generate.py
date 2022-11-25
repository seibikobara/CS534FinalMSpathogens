#!/usr/bin/env python3_9

import numpy as np
import pandas as pd
import os.path
import tracemalloc

'''
# data 
* DRIAMS has 2015-2018 for region A, B .C. D. The original study used DRIAMS-A, so that this project use data of DRIAMS-A in 2015 to 2018.

# pathogen
E-coli
A,B,C,D all use Escherichia coli for E-coli. no such e-coli, etc.

Klebssiella pneumoniae
several specices, Klebsiella pneumoniae, Klebsiella oxytoca, ect.
Use only pneumoniaw

staphyolococcus aureus 
# Used code for check
ID = pd.read_csv("~/projects/pathogen_ms/DRIAMS-A/id/2018/2018_clean.csv")
species = pd.DataFrame(ID["species"].value_counts())
temp = species.index
index = temp.str.contains('^(Escherichia coli)$',regex=True,case=False)
temp[index]


'''
tracemalloc.start()

pathogen = 'Staphylococcus aureus'
key_ = 'staph'
years = ['2015','2016','2017','2018']
susc_data = pd.DataFrame()
intens_data = pd.DataFrame()
for j in years:
    id_path = '/Users/seibi/emory/cs534/CS534FinalMSpathogens/DRIAMS-A/id/' + j + '/' + j + '_clean.csv'
    ID = pd.read_csv(id_path)
    contain = '^(' + pathogen + ')$'
    index = ID['species'].str.contains(contain,regex=True,case=False)
    pathogens_files = ID.loc[index,:]
    file_names = pathogens_files["code"].tolist()
    included_file_name = list()
    # collect MS data
    all_data = list()
    for i in np.arange(0,len(file_names),1):
        file_ = file_names[i]
        filepath = '/Users/seibi/emory/cs534/CS534FinalMSpathogens/DRIAMS-A/binned_6000/' + j + '/' + file_+'.txt'
        print("=============================")
        number = str(i)
        text = 'currently in the year '+j+' for ith file: '+number
        memory = str(tracemalloc.get_traced_memory())
        memory_text = 'current memory: '+ memory
        print(text)
        print(memory_text)
        print("=============================")    
        if os.path.exists(filepath):
            data = pd.read_csv(filepath,sep=' ')
            data.columns = ['bin',file_]
            data_intensity = data.iloc[:,1]
            all_data = all_data + [data_intensity]
            included_file_name = included_file_name + [file_]
        else: 
            pass
    ms_res = pd.DataFrame(all_data, index = included_file_name)
    included_susc = ID.loc[ID['code'].isin(included_file_name),:]
    # for final output
    intens_data = pd.concat([intens_data, ms_res], axis = 0)
    susc_data = pd.concat([susc_data, included_susc], axis = 0)
        

save_ms_name = key_ + '_ms.csv'
save_susc_name = key_ + '_susceptibility.csv'
intens_data.to_csv(save_ms_name)
susc_data.to_csv(save_susc_name)
print("=============================")
memory = str(tracemalloc.get_traced_memory())
memory_text = 'memory at saving : '+ memory
print(text)
print(memory_text)
print("=============================")    
tracemalloc.stop()