#!bin/zsh python 3.9


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
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, Flatten, MaxPooling1D,MaxPooling2D



# CNN ID

x = pd.read_csv("cipro_x.csv")
y = pd.read_csv('cipro_y.csv')

# reshape for CNN
# https://www.datatechnotes.com/2020/02/classification-example-with-keras-cnn.html
array = x.to_numpy()
array2 = array.reshape(array.shape[0], array.shape[1], 1)
print(x.shape)
x_train, x_test, y_train, y_test=train_test_split(array2, y, test_size=0.3)

model = Sequential()
model.add(Conv1D(64, 2, activation="relu", input_shape=(6000,1)))
model.add(Dense(16, activation="relu"))
model.add(MaxPooling1D())
model.add(Flatten())
model.add(Dense(2, activation = 'sigmoid'))
model.compile(loss = 'sparse_categorical_crossentropy', 
     optimizer = "adam",               
              metrics = ['accuracy'])
model.summary()
model.fit(x_train, y_train, batch_size=16,epochs=2, verbose=0)

y_prob = model.predict(x_test)[:,1]
y_pred = y_prob.argmax(axis=-1)
y_true = y_test

roc =  roc_auc_score(y_true, y_prob)# 0.5226



# CNN2D
array3= array.reshape(array.shape[0], 100, 60,1)
x_train, x_test, y_train, y_test=train_test_split(array3, y, test_size=0.3)

model = Sequential()
model.add(Conv2D(32, (1,1), activation="relu", input_shape=(100,60,1)))
model.add(Dense(128, activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(2, activation = 'sigmoid'))
model.compile(loss = 'sparse_categorical_crossentropy', 
     optimizer = "adam", metrics = ['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=16,epochs=2, verbose=0)

y_prob = model.predict(x_test)[:,1]
y_pred = y_prob.argmax(axis=-1)
y_true = y_test

roc =  roc_auc_score(y_true, y_prob) # 0.5598
