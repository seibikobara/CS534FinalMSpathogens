

'''
Staphylococcus aureus
Target drugs:     
    |--Ciprofloxacin
    |--Fusidic acid
    |--Oxacillin
Outcome 1 : resistance, intermediate. 0: susceotible
'''

import numpy as np
import pandas as pd
import os.path
import tracemalloc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.neural_network import MLPClassifier

# split
# train, validation, test
def split_train_val_test(data_x, data_y):
    x_train_temp, x_test, y_train_temp, y_test = train_test_split(data_x,data_y,test_size = 0.3, random_state=25)
    # train-> train + validation
    x_train, x_val, y_train, y_val = train_test_split(x_train_temp, y_train_temp, test_size = 0.3, random_state=25)
    return x_train, x_val, x_test, y_train, y_val, y_test


def mlp_wo_Smote(features, outcome):
    acc_list=list()
    roc_list = list()
    skf = StratifiedKFold(n_splits=5)
    #smote = SMOTE(random_state = 1, k_neighbors=6)
    #x_data, y_data = smote.fit_resample(x_data, y_data)
    for train_index, test_index in skf.split(features,outcome):
        x_train, x_test = x.iloc[train_index,:], x.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # model 
        mlpc = MLPClassifier(solver='lbfgs', alpha=0.001,  hidden_layer_sizes=(30, 10), random_state=1)
        mlpc.fit(x_train, y_train)
        # predict
        y_pred = mlpc.predict(x_test)
        y_true = y_test
        # accuracy
        acc = accuracy_score(y_true, y_pred)
        acc_list = acc_list + [acc]
        # probability
        y_prob = mlpc.predict_proba(x_test)[:, 1]
        # roc
        roc = roc_auc_score(y_true, y_prob)
        roc_list = roc_list + [roc]
    #fig, axs = plt.subplots(1, 2)
    # accuracy
    #acc_mean = np.array(acc_list).mean()
    #acc_sd = np.array(acc_list).std()
    #acc_text = 'Accuracy: %.4f (%.4f)' %(acc_mean, acc_sd)
    #create boxplot
    #axs[0].boxplot(acc_list) 
    #axs[0].set_title(acc_text, fontsize = 8)
    #axs[0].set_ylim([0,1])
    # roc
    #roc_mean = np.array(roc_list).mean()
    #roc_sd = np.array(roc_list).std()
    #roc_text = 'AUROC: %.4f (%.4f)' %(roc_mean, roc_sd)
    #create boxplot
    #axs[1].boxplot(roc_list)
    #axs[1].set_ylim([0,1]) 
    #axs[1].set_title(roc_text, fontsize=8)

    return roc_list



def mlp_Smote(features, outcome):
    acc_list=list()
    roc_list = list()
    skf = StratifiedKFold(n_splits=5)
    smote = SMOTE(random_state = 1, k_neighbors=6)
    x, y = smote.fit_resample(features, outcome)
    for train_index, test_index in skf.split(x,y):
        x_train, x_test = x.iloc[train_index,:], x.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # model 
        mlpc = MLPClassifier(solver='lbfgs', alpha=0.001,  hidden_layer_sizes=(30, 10), random_state=1)
        mlpc.fit(x_train, y_train)
        # predict
        y_pred = mlpc.predict(x_test)
        y_true = y_test
        # accuracy
        acc = accuracy_score(y_true, y_pred)
        acc_list = acc_list + [acc]
        # probability
        y_prob = mlpc.predict_proba(x_test)[:, 1]
        # roc
        roc = roc_auc_score(y_true, y_prob)
        roc_list = roc_list + [roc]
    #fig, axs = plt.subplots(1, 2)
    # accuracy
    #acc_mean = np.array(acc_list).mean()
    #acc_sd = np.array(acc_list).std()
    #acc_text = 'Accuracy: %.4f (%.4f)' %(acc_mean, acc_sd)
    #create boxplot
    #axs[0].boxplot(acc_list) 
    #axs[0].set_title(acc_text, fontsize = 8)
    #axs[0].set_ylim([0,1])
    # roc
    #roc_mean = np.array(roc_list).mean()
    #roc_sd = np.array(roc_list).std()
    #roc_text = 'AUROC: %.4f (%.4f)' %(roc_mean, roc_sd)
    #create boxplot
    #axs[1].boxplot(roc_list)
    #axs[1].set_ylim([0,1]) 
    #axs[1].set_title(roc_text, fontsize=8)
    return roc_list



drugs = ['cipro','fusidic','oxa']
res = list()
for i in drugs:
    xfilename_ = i + '_x.csv'
    yfilename_ = i + '_y.csv'
    x = pd.read_csv(xfilename_)
    y = pd.read_csv(yfilename_)
    roc = mlp_wo_Smote(x,y)
    res = res + [roc]
temp = pd.DataFrame(res)
result = temp.T
result.columns = drugs
result.to_csv('roc_staphy_mlp.txt', sep='\t',index=False)


drugs = ['cipro','fusidic','oxa']
res = list()
for i in drugs:
    xfilename_ = i + '_x.csv'
    yfilename_ = i + '_y.csv'
    x = pd.read_csv(xfilename_)
    y = pd.read_csv(yfilename_)
    roc = mlp_Smote(x,y)
    res = res + [roc]
temp = pd.DataFrame(res)
result_smote = temp.T
result_smote.columns = drugs
result_smote.to_csv('roc_staphy_mlp_smote.txt', sep='\t',index=False)
