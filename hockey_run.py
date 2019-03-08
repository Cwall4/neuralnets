# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 14:04:11 2017

@author: Colin

This file loads player data, drops observations with any missing data,
scales the data, details a neural network model, and fits the model.
The model with the lowest validation mean absolute error is saved as 'player_model.hdf5'.
The model can then predict retirement age when given data. Note that input data should be scaled,
and output data should be unscaled to ease interpretation of results.

Go has a package for hdf5 here: https://godoc.org/bitbucket.org/binet/go-hdf5/pkg/hdf5
So you should be able to open the model and get predictions in Go.
As for refitting the model, you may want to keep this in Python, and refit after every season,
since more (especially recent) observations should help accuracy.
Even the structure of the model may be worth changing if estimates improve.
"""

import hockey_loader
import network2
#import pybrain
#import keras
import numpy as np
np.set_printoptions(threshold=np.inf)
from sklearn import preprocessing, metrics
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import Dense, Activation, normalization, Dropout
from keras.optimizers import SGD, RMSprop, Nadam
from keras import regularizers, initializers, utils
from keras.callbacks import ModelCheckpoint
import pydot_ng as pydot
#import h5py
#print(pydot.find_graphviz())

data = hockey_loader.load_data()

data = data[~np.isnan(data).any(axis=1)]

print(data.shape)

#print(data[:,np.r_[:18,22:23]])

#test = np.random.random((1000, 100))

scaler = preprocessing.StandardScaler().fit(data)

print(scaler.scale_)

data = scaler.transform(data)

#data_in = data[:,np.r_[:18,22:24]]

data_in = data[:,np.r_[:18, 22:25]]

pca = PCA()

pca.fit(data_in)

params = pca.get_params()

print('PCA components \n%s' %pca.components_)

PCAvar = pca.explained_variance_

print('PCA Explained variance \n%s' %pca.explained_variance_ratio_)

"""

The first 14 PCA variables explain a total of almost 99 percent of variance.

"""

print(sum(pca.explained_variance_ratio_[0:14]))

data_inPCA = data_in[:,0:11]

#print(data_in.names)

nin = data_in.shape[1]

ninPCA = data_inPCA.shape[1]

model = Sequential()

lmbda = 1e-3

inits = initializers.glorot_normal()

#model.add(Dense(54, activation = 'tanh', kernel_initializer = inits, input_shape=(nin,), kernel_regularizer = regularizers.l2(lmbda)))
#model.add(Dropout(0.1))
model.add(Dense(36, activation = 'tanh', kernel_initializer = inits, input_shape=(nin,), kernel_regularizer = regularizers.l2(lmbda)))
#model.add(Dense(36, activation = 'tanh', kernel_initializer = inits, kernel_regularizer = regularizers.l2(lmbda)))
model.add(Dropout(0.1))
model.add(Dense(18, activation = 'tanh', kernel_initializer = inits, kernel_regularizer = regularizers.l2(lmbda)))
model.add(Dense(9, activation = 'softmax', kernel_initializer = inits, kernel_regularizer = regularizers.l2(lmbda)))
model.add(Dense(1))

"""

Model2 currently looks like it returns the lowest error.
PCA, then providing the first ~12 columns that explain the most variance doesn't seem to improve results.
Maybe PCA while still providing all columns would help?

"""

model2 = Sequential()

lmbda2 = 1e-3

model2.add(Dense(36, activation = 'relu', kernel_initializer = inits, input_shape=(nin,), kernel_regularizer = regularizers.l2(lmbda2)))
model2.add(Dropout(0.1))
#model2.add(Dense(36, activation = 'tanh', kernel_initializer = inits, kernel_regularizer = regularizers.l2(lmbda)))
model2.add(Dense(18, activation = 'relu', kernel_initializer = inits, kernel_regularizer = regularizers.l2(lmbda2)))
model2.add(Dropout(0.1))
model2.add(Dense(9, activation = 'relu', kernel_initializer = inits, kernel_regularizer = regularizers.l2(lmbda2)))
#model2.add(Dropout(0.1))
#model2.add(Dense(9, activation = 'relu', kernel_initializer = inits, kernel_regularizer = regularizers.l2(lmbda2)))
model2.add(Dense(1))

model3 = Sequential()

lmbda3 = 1e-3

model3.add(Dense(18, activation = 'tanh', kernel_initializer = inits, input_shape=(ninPCA,), kernel_regularizer = regularizers.l2(lmbda3)))
model3.add(Dropout(0.1))
#model3.add(Dense(15, activation = 'tanh', kernel_initializer = inits, kernel_regularizer = regularizers.l2(lmbda3)))
#model3.add(Dropout(0.1))
model3.add(Dense(12, activation = 'tanh', kernel_initializer = inits, kernel_regularizer = regularizers.l2(lmbda3)))
#model3.add(Dropout(0.1))
#model3.add(Dense(9, activation = 'tanh', kernel_initializer = inits, kernel_regularizer = regularizers.l2(lmbda3)))
model3.add(Dropout(0.1))
model3.add(Dense(6, activation = 'tanh', kernel_initializer = inits, kernel_regularizer = regularizers.l2(lmbda3)))
model3.add(Dense(1))

# Rectified linear activations seem somewhat effective here, reaching val_mae around 0.4580 at some points.

sgd = SGD(lr=0.005, momentum = 0.9, decay = 1e-6, nesterov = True)

rmsprop = RMSprop()

adap = Nadam(lr = 0.002)

model.compile(optimizer=adap,
              loss='mean_squared_error',
              metrics=['mae'])

model2.compile(optimizer=adap,
              loss='mean_squared_error',
              metrics=['mae'])

model3.compile(optimizer=adap,
              loss='mean_squared_error',
              metrics=['mae'])

"""

These checkpointers save the neural network being fitted,

overwriting the previous if it has improved based on the given metric (the "monitor" variable).

"""

checkpointer2 = ModelCheckpoint(filepath="player_model2.hdf5", monitor = 'val_mean_absolute_error', mode = 'min', verbose=1, save_best_only=True)

checkpointer3 = ModelCheckpoint(filepath="player_model3.hdf5", monitor = 'val_mean_absolute_error', mode = 'min', verbose=1, save_best_only=True)

"""

Decreasing nepochs is an immediate way to speed up fitting time.

I had it saved at 40, but most networks I'm trying seem to level off around 10 to 20 epochs.

"""

nepochs = 5

hist = model.fit(data_in, data[:,21], epochs = nepochs, batch_size = 32, validation_split = 0.1)

hist2 = model2.fit(data_in, data[:,21], epochs = nepochs, batch_size = 32, validation_split = 0.1, callbacks=[checkpointer2])

hist3 = model3.fit(data_inPCA, data[:,21], epochs = nepochs, batch_size = 32, validation_split = 0.1, callbacks=[checkpointer3])

#utils.plot_model(model)

# The best results I had were a validation mae of ~0.4280.

#print(model.get_weights())

age_subset = data_in[:,2] == (24 - scaler.mean_[2]) / scaler.scale_[2]

retAge_pred = scaler.mean_[21] + scaler.scale_[21] * model.predict(data_in[age_subset,:], verbose = 1)

retAge_pred2 = scaler.mean_[21] + scaler.scale_[21] * model2.predict(data_in[age_subset,:], verbose = 1)

retAge_pred3 = scaler.mean_[21] + scaler.scale_[21] * model3.predict(data_inPCA[age_subset,:], verbose = 1)

retAge_obs = scaler.mean_[21] + scaler.scale_[21] * data[age_subset,21]

print(metrics.mean_absolute_error(retAge_obs, retAge_pred), metrics.mean_absolute_error(retAge_obs, retAge_pred2), metrics.mean_absolute_error(retAge_obs, retAge_pred3), sep = ', ')