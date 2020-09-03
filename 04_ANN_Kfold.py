# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 12:27:59 2020

@author: boehm

Setup 04 FFNN_raw4 - kfold
Training on CDS + 52 hist. Events; shifted / multiplied CDS events; 154 training samples
"""

import os
import numpy as np
import pandas as pd
#%%############# Data import #################################################
#import rain array ###########################################################
array_rain_hist_1 = np.load(r'C:\Users\boehm\Google_Drive\Masterarbeit\Part4_ANN_versions\hist_rain_data\dfs0\1_5_factor_13h\final_rain_array\rain_series_hist_1_5.npy')
array_rain_hist_2 = np.load(r'C:\Users\boehm\Google_Drive\Masterarbeit\Part4_ANN_versions\hist_rain_data\dfs0\3_factor_13h\final_rain_array\rain_series_hist_3.npy')
array_rain_CDS = np.load(r'C:\Users\boehm\Google_Drive\Masterarbeit\Part4_ANN_versions\CDS_rain_data\CDS_rain_series\rain_series_CDS_uneven\rain_series.npy')


#%% import flood arrays #########################################################
##### CDS ################
os.chdir(r'C:\Users\boehm\Google_Drive\Masterarbeit\Part4_ANN_versions\CDS_rain_data\CDS_flood_maps\Numpy-files_CDS_uneven')
path = r'C:\Users\boehm\Google_Drive\Masterarbeit\Part4_ANN_versions\CDS_rain_data\CDS_flood_maps\Numpy-files_CDS_uneven'
files = os.listdir(path)
#print(files)

flood_arrays_CDS = {}

for file in files:
    flood_arrays_CDS[file] = np.load(file)
    
data=list(flood_arrays_CDS.values())
del flood_arrays_CDS, files, file, path
array_flood_CDS = np.asarray(data)
del data

array_flood_CDS = np.reshape(array_flood_CDS,(21,-1))
array_flood_CDS = np.concatenate((array_flood_CDS,array_flood_CDS,array_flood_CDS,array_flood_CDS,array_flood_CDS)) # create array matching input
##### hist 1 #################
os.chdir(r'C:\Users\boehm\Google_Drive\Masterarbeit\Part4_ANN_versions\hist_rain_data\dfs2\1_5_factor_13h\flood_maps')
path = r'C:\Users\boehm\Google_Drive\Masterarbeit\Part4_ANN_versions\hist_rain_data\dfs2\1_5_factor_13h\flood_maps'
files = os.listdir(path)

files = sorted(files, key=lambda item: int(item.split('_')[1].split('.')[0]))
print(files)

flood_arrays_hist = {}

for file in files:
    flood_arrays_hist[file] = np.load(file)
    
data=list(flood_arrays_hist.values())
del flood_arrays_hist, files, file, path
array_flood_hist_1 = np.asarray(data)
del data
array_flood_hist_1 = np.reshape(array_flood_hist_1,(24,-1))
##### hist 2 #################
os.chdir(r'C:\Users\boehm\Google_Drive\Masterarbeit\Part4_ANN_versions\hist_rain_data\dfs2\3_factor_13h\flood_maps')
path = r'C:\Users\boehm\Google_Drive\Masterarbeit\Part4_ANN_versions\hist_rain_data\dfs2\3_factor_13h\flood_maps'
files = os.listdir(path)

files = sorted(files, key=lambda item: int(item.split('_')[1].split('.')[0]))
print(files)

flood_arrays_hist = {}

for file in files:
    flood_arrays_hist[file] = np.load(file)
    
data=list(flood_arrays_hist.values())
del flood_arrays_hist, files, file, path
array_flood_hist_2 = np.asarray(data)
del data
array_flood_hist_2 = np.reshape(array_flood_hist_2,(28,-1))
#%% zero padding of CDS input
pad_after = np.pad(array_rain_CDS,((0,0),(0,66)),'constant')
pad_before = np.pad(array_rain_CDS,((0,0),(57,9)),'constant') # 1,5 h buffer before end -> time of concentration
pad_middle = np.pad(array_rain_CDS,((0,0),(33,33)),'constant')
pad_before_middle = np.pad(array_rain_CDS,((0,0),(20,46)),'constant')
pad_after_middle = np.pad(array_rain_CDS,((0,0),(46,20)),'constant')

array_rain_CDS = np.concatenate((pad_before,pad_before_middle,pad_middle,pad_after_middle,pad_after))
del pad_after, pad_before, pad_middle, pad_after_middle,pad_before_middle
#%% seperate single hist event for prediction, merge CDS and hist events
X = np.concatenate((array_rain_hist_1,array_rain_hist_2,array_rain_CDS))
y = np.concatenate((array_flood_hist_1,array_flood_hist_2,array_flood_CDS))

# X_pred = X[11]
# X_pred = np.reshape(X_pred,(1,78))
# y_sim = y[11]
# y_sim = np.reshape(y_sim,(1,361315))

# X = np.delete(X,[11], axis = 0) # delete holdback
# y = np.delete(y,[11], axis = 0)

del array_flood_CDS, array_flood_hist_1, array_flood_hist_2, array_rain_CDS, array_rain_hist_1, array_rain_hist_2
#%%
from sklearn.model_selection import KFold # k-fold cross validation
num_folds = 10 # K = 10

kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
#validation_split = 0.1

score = [] 
oos_y = []
oos_pred = []
epochs_needed = []

#%% model
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn import metrics
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# determine the number of input features:
n_features = X.shape[1] #or X

plt.figure()
# define model in for loop for cross validation:
fold = 0
for train, test in kfold.split(X, y):
    fold+=1 # Increase fold number
# generate print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold} ...')
    model = Sequential()
    model.add(Dense(78, activation='relu', kernel_initializer='he_normal',
                    input_shape=(n_features,)))
    model.add(Dropout(0.2))
    model.add(Dense(78, activation='tanh', kernel_initializer='he_normal'))
    model.add(Dense(361315))
    

# compile the model:
    #optimizer = tf.keras.optimizers.RMSprop(0.001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False, name='Adam')
    model.compile(optimizer=optimizer,loss='mean_squared_error')
    
# early stopping:
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=8,verbose=1, mode='auto',restore_best_weights=True)
# fit the model:
    history = model.fit(X[train], y[train],epochs=200, batch_size=32, validation_data = (X[test],y[test]), verbose=1, callbacks=[monitor])

# out of sample prediction
    yhat = model.predict(X[test])
# Measure this fold's RMSE

    #rmse  = np.sqrt(metrics.mean_squared_error(yhat, y_sim)) # sqrt over all RMSE
    #print(f"Fold score (RMSE): {score}")
    

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'],'--')
    #plt.legend(['train', 'test'], loc='upper left') 

    #score.append(rmse)
    oos_pred.append(yhat)
    oos_y.append(y[test])
    
    epochs = monitor.stopped_epoch
    epochs_needed.append(epochs)

# save model
    model.save(os.path.join(r'C:\Users\boehm\Google_Drive\Masterarbeit\Part4_ANN_versions\ANNs\Models\04_FFNN',f'model{fold}_epochs_earlystop_dp02'))

plt.title(f'Model loss fold-{fold} - Kfold (K=10)')
plt.ylabel('loss')
plt.xlabel('epoch')
   
plt.show()
    
#epoch:iteration over samples;batch_size: total number of training examples present in one batch;verbose:progress bar (off/on)

# evaluate the model:
#    scores_all_test = model.evaluate(X[test], y[test], verbose=1)

