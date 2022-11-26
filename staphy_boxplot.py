import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


lr = pd.read_csv("roc_staphy_lr.txt",sep = '\t')
lr_smote = pd.read_csv("roc_staphy_lr_smote.txt",sep = '\t')
lr_full = pd.concat([lr, lr_smote], axis = 1) 

lgbm = pd.read_csv("roc_staphy_lgb.txt",sep = '\t')
lgbm_smote = pd.read_csv("roc_staphy_lgb_smote.txt",sep = '\t')
lgbm_full = pd.concat([lgbm, lgbm_smote], axis = 1) 

mlp =pd.read_csv("roc_staphy_mlp.txt",sep = '\t')
mlp_smote = pd.read_csv("roc_staphy_mlp_smote.txt",sep = '\t')
mlp_full = pd.concat([mlp, mlp_smote], axis = 1) 

cnn1d =pd.read_csv("roc_staphy_cnn1d.txt",sep = '\t')
cnn1d_smote = pd.read_csv("roc_staphy_cnn1d_smote.txt",sep = '\t')
cnn1d_full = pd.concat([cnn1d, cnn1d_smote], axis = 1) 

cnn2d = pd.read_csv("roc_staphy_cnn2d.txt",sep = '\t')
cnn2d_smote = pd.read_csv("roc_staphy_cnn2d_smote.txt",sep = '\t')
cnn2d_full = pd.concat([cnn2d, cnn2d_smote], axis = 1) 




labels = ['No-SMOTE','SMOTE']
titles = ['Ciprofloxacin','Fusidic acid','Oxacillin']
fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (5,5)) 
for i, ax in enumerate(axs.flat):
    ax.boxplot(lr_full.iloc[:,[i,i+3]])
    ax.set_title(titles[i], fontsize = 12)
    ax.set_ylim([0,1])
    ax.set_xticklabels(labels, fontsize=8)


labels = ['No-SMOTE','SMOTE']
titles = ['Ciprofloxacin','Fusidic acid','Oxacillin']
fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (5,5)) 
for i, ax in enumerate(axs.flat):
    ax.boxplot(lgbm_full.iloc[:,[i,i+3]])
    ax.set_title(titles[i], fontsize = 12)
    ax.set_ylim([0,1])
    ax.set_xticklabels(labels, fontsize=8)


labels = ['No-SMOTE','SMOTE']
titles = ['Ciprofloxacin','Fusidic acid','Oxacillin']
fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = (5,5)) 
for i, ax in enumerate(axs.flat):
    ax.boxplot(mlp_full.iloc[:,[i,i+3]])
    ax.set_title(titles[i], fontsize = 12)
    ax.set_ylim([0,1])
    ax.set_xticklabels(labels, fontsize=8)


