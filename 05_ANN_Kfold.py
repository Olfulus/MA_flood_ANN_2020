# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:31:29 2020

@author: boehm
"""

import os
import numpy as np
#%%############# Data import #################################################
#import simple rain input ###########################################################
array_rain_hist_1 = np.load(r'C:\Users\boehm\Google_Drive\Masterarbeit\Part4_ANN_versions\hist_rain_data\dfs0\1_5_factor_13h\hist_1_5fct_30_60_90_accum.npy')
array_rain_hist_2 = np.load(r'C:\Users\boehm\Google_Drive\Masterarbeit\Part4_ANN_versions\hist_rain_data\dfs0\3_factor_13h\hist_3fct_30_60_90_accum.npy')
array_rain_CDS = np.load(r'C:\Users\boehm\Google_Drive\Masterarbeit\Part4_ANN_versions\CDS_rain_data\CDS_rain_series\rain_series_CDS_uneven\CDS_uneven_30_60_90_accum.npy')


#%%
#import flood arrays #########################################################
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
##### hist #################
os.chdir(r'C:\Users\boehm\Google_Drive\Masterarbeit\Part4_ANN_versions\hist_rain_data\dfs2\1_5_factor_13h\flood_maps')
path = r'C:\Users\boehm\Google_Drive\Masterarbeit\Part4_ANN_versions\hist_rain_data\dfs2\1_5_factor_13h\flood_maps'
files = os.listdir(path)
#print(files)

files = sorted(files, key=lambda item: int(item.split('_')[1].split('.')[0])) 

flood_arrays_hist_1 = {}

for file in files:
    flood_arrays_hist_1[file] = np.load(file)

data=list(flood_arrays_hist_1.values())
del flood_arrays_hist_1, files, file, path
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
#%%
X = np.concatenate((array_rain_CDS,array_rain_hist_1))
y = np.concatenate((array_flood_CDS,array_flood_hist_1))

# normalize data
X_mean = np.mean(X)
X_std = np.std(X)
X = (X - X_mean) / X_std

y_mean = np.mean(y)
y_std = np.std(y)
y = (y - y_mean) /  y_std

#X_val = array_rain_hist_2
#y_val = array_flood_hist_2

#%%
from sklearn.model_selection import KFold # k-fold cross validation
num_folds = 10 # K = 10

kfold = KFold(n_splits=num_folds, shuffle=True, random_state=42)
#validation_split = 0.1

score = [] 
oos_y = []
oos_pred = []
epochs_needed = []

#%%
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
    model.save(os.path.join(r'C:\Users\boehm\Google_Drive\Masterarbeit\Part4_ANN_versions\ANNs\Models\05_FFNN',f'model{fold}_epochs_earlystop_dp02'))

plt.title('Model loss - Kfold (K=10)')
plt.ylabel('loss')
plt.xlabel('epoch')
   
plt.show()

#epoch:iteration over samples;batch_size: total number of training examples present in one batch;verbose:progress bar (off/on)

# evaluate the model:
    #scores = model.evaluate(inputs[test], targets[test], verbose=1)
    
#%% build the oos prediction list and calculate the error
oos_y = np.concatenate(oos_y)
oos_pred = np.concatenate(oos_pred)
#%%
score_final = np.sqrt(metrics.mean_squared_error(oos_pred, oos_y))
print(f"Final, out of samples score (RMSE): {score_final}")
#%% 
# #%%
# make a prediction
# X_event = array_rain_hist[11]
# X_event = np.reshape(X_event,(1,4))
# y_event = array_flood_hist[11]
# y_event = np.reshape(y_event,(1,361315))

yhat = model.predict([X_pred])
#print('Predicted: %.3f' % yhat_reshape)

index_yhat = np.where(yhat>0.01,yhat,0)
index_y_event = np.where(y_sim>0.01,y_sim,0)
pred_RMSE = metrics.mean_squared_error(index_yhat, index_y_event) #wrong because rmse dependent on n cells

import matplotlib.pyplot as plt
import matplotlib.lines as lines

line = lines.Line2D([0, 1], [0, 1], color='red')

fig, ax = plt.subplots() #
ax.scatter(index_yhat, index_y_event, cmap='Blues', vmin= 0, vmax = 2)
plt.plot([0, 0], [1, 1], 'k-', color = 'r')
ax.add_line(line)
plt.xlabel("Prediction")
plt.ylabel("Hydraulic Model")
plt.title('Historical event 11')

#plot prediction and simulation
yhat_re = np.reshape(yhat,(569,635))

plt.figure(2) #prediction
plt.imshow(yhat_re, cmap='Blues', interpolation='nearest',vmin=0, vmax=0.8)
plt.colorbar()
plt.title('EV_11 prediction')

CDS45_map = np.reshape(y_sim,(569,635))

plt.figure(3) #simulation
plt.imshow(CDS45_map,cmap='Blues', interpolation='nearest',vmin=0, vmax=0.8)
plt.colorbar()
plt.title('EV_11 simulation')
plt.show()

#%% confusion matrix
 
# create classes for regression output
y_sim_classes = np.zeros_like(y_sim)    # initialise a matrix full with zeros
y_sim_classes[y_sim > 0.01] = 1
y_sim_classes[y_sim > 0.1] = 2 
y_sim_classes[y_sim > 0.25] = 3
y_sim_classes[y_sim > 0.5] = 4
y_sim_classes[y_sim > 1] = 5
y_sim_classes = np.reshape(y_sim_classes,(361315,-1))

yhat_classes = np.zeros_like(yhat)
yhat_classes[yhat > 0.01] = 1
yhat_classes[yhat > 0.1] = 2
yhat_classes[yhat > 0.25] = 3
yhat_classes[yhat > 0.5] = 4
yhat_classes[yhat > 1] = 5
yhat_classes = np.reshape(yhat_classes,(361315,-1))

# confusion matrix for all values
matrix_con = metrics.confusion_matrix(y_sim_classes,yhat_classes,labels=[0,1,2,3,4,5])#.ravel()
# confusion matrix for values above 1 cm
matrix_con_without_0 = metrics.confusion_matrix(y_sim_classes,yhat_classes,labels=[1,2,3,4,5])

cm_display = metrics.ConfusionMatrixDisplay(matrix_con,display_labels=[0,1,2,3,4,5]).plot(2)

y_sim_classes_map = np.reshape(y_sim_classes,(569,635))
plt.figure(14) #simulation
plt.imshow(y_sim_classes_map,cmap='Greys',vmin=0, vmax=5)
plt.colorbar()
plt.title('y_sim_classes_map')
plt.show()

yhat_classes_map = np.reshape(yhat_classes,(569,635))
plt.figure(15) #simulation
plt.imshow(yhat_classes_map,cmap='Greys',vmin=0, vmax=5)
plt.colorbar()
plt.title('yhat_classes_map')
plt.show()



misses = sum(matrix_con[:,0])-matrix_con[0,0]
false_alarm = sum(matrix_con[0])-matrix_con[0,0]
hits = matrix_con.sum()-misses-false_alarm-matrix_con[0,0]


csi_value =  hits/(hits+misses+false_alarm)#does not take into account different depths



#%% Hit-miss map
# differenz in models
y_diff = y_sim_classes_map - yhat_classes_map
plt.figure(16) #simulation
plt.imshow(y_diff,cmap='Greys',vmin=0, vmax=5)
plt.colorbar()
plt.title('y_diff_map')
plt.show()


# create hit miss matrix
y_diff_2 = np.zeros_like(y_sim_classes)  
y_diff_2 = np.where(np.logical_and(y_sim_classes == 0, yhat_classes == 0),1,y_diff_2) #no flooding in both models
y_diff_2 = np.where(np.logical_and(y_sim_classes > 0, yhat_classes > 0),2,y_diff_2) #HIT: flooding in both models
y_diff_2 = np.where(np.logical_and(y_sim_classes > 0, yhat_classes == 0),3,y_diff_2) #MISS: flooding in hydraulic but not in yhat
y_diff_2 = np.where(np.logical_and(y_sim_classes == 0, yhat_classes > 0),4,y_diff_2) #FALSE POSITIVE: flooding in yhat but not in hydraulic

from matplotlib.colors import ListedColormap
import seaborn as sns
y_diff_2_map = np.reshape(y_diff_2,(569,635))

cmap = ListedColormap(["whitesmoke", "green", "red", "gold"])

plt.figure(19)

ax = plt.axes()
sns.heatmap(y_diff_2_map, ax = ax, cmap=cmap,yticklabels=False,xticklabels=False)
ax.set_title('Hit-miss map')

plt.figure(17) #simulation
plt.imshow(y_diff_2_map, cmap=cmap, vmin=0, vmax=4)
plt.colorbar()
plt.title('Hit-miss map')
plt.show()
