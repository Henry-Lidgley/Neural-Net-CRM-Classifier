#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# @author: henrylidgley 2018

# ACI Project ANN Classifiers - File 2 of 2

# Import APIs and methods

from __future__ import print_function

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.utils import compute_class_weight

from keras.models import Sequential
from keras.callbacks import History, EarlyStopping, ModelCheckpoint 
from keras.layers.advanced_activations import PReLU
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import  nadam, adam, sgd

from hyperopt import Trials, STATUS_OK, tpe, hp, fmin

# For adding new activation function
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

# Set classifier params (upselling, appetency, churn)
target = 'churn'
# Set prediction probability threshold
pred_threshold = 0.5
# Set number of hyperparameter search evaluations
evaluations = 50

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

# Make swish activation function
def swish(x):
    return (K.sigmoid(x) * x)

# Set optimizer params
nadam = nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
adam = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.004, amsgrad=False)
sgd = sgd(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Create activation function method
def activation_func(name):
    if name == 'swish':
        get_custom_objects().update({'swish': Activation(swish)})
        return Activation(swish)
    if name == 'PReLU':
        return PReLU()
    if name == 'relu':
        return Activation('relu')
    
# Import, split and standardise dataset depending on target   
def data(target):
    
    X = pd.read_csv('orangeXdataNaN=0.csv', sep='\t')
    X = X.iloc[:, 1:388].values

    upselling = pd.read_csv("orange_small_train_upselling.labels", sep='\t', names=['Upselling'])
    upselling[upselling==-1] = 0

    appetency = pd.read_csv("orange_small_train_appetency.labels", sep='\t', names=['Appetency'])
    appetency[appetency==-1] = 0

    churn = pd.read_csv("orange_small_train_churn.labels.txt", sep='\t', names=['Churn'])
    churn[churn==-1] = 0

    Y = pd.concat([upselling, appetency, churn], axis=1)
    Y.columns = ['upselling', 'appetency', 'churn']

    from sklearn.model_selection import train_test_split
    # Create train and test data    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    # Create val data
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.25, random_state = 0)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform(X_val)
    X_test = sc.transform(X_test)

    del appetency, churn, upselling, X, Y
       
    if (target == 'upselling'):
        Y_train = Y_train.iloc[:, 0].values
        Y_test = Y_test.iloc[:, 0].values
        Y_val = Y_val.iloc[:, 0].values
    
    if (target == 'appetency'):
        Y_train = Y_train.iloc[:, 1].values
        Y_test = Y_test.iloc[:, 1].values
        Y_val = Y_val.iloc[:, 1].values
    
    if (target == 'churn'):
        Y_train = Y_train.iloc[:, 2].values
        Y_test = Y_test.iloc[:, 2].values
        Y_val = Y_val.iloc[:, 2].values
        
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    X_test = X_test.astype('float32')

    return X_train, Y_train, X_test, Y_test, X_val, Y_val

X_train, Y_train, X_test, Y_test, X_val, Y_val = data(target)

# Create hyperparameter search space
space = {'choice': hp.choice('num_layers',
                    [ {'layers':'one', },
                    {'layers':'two',
                    'units2': hp.choice('units2', [20,30,40,50,60,70,80,90]), 
                    'dropout2': hp.uniform('dropout2', .2,.5)}
                    ]),

            'units1': hp.choice('units1', [20,30,40,50,60,70,80,90]),
            'dropout1': hp.uniform('dropout1', .2,.5),
            'batch_size': hp.choice('batch_size', [16,32,64,128]),

            'optimizer': hp.choice('optimizer',['nadam', 'adam', 'sgd']),
            'activation_func': hp.choice('activation_func',['swish', 'PReLU', 'relu']),
        }

# Instantiate, train and test models   
def f_nn(params):   

    print ('Params testing: ', params)
    
    # Create model
    model = Sequential()
    
    # Create first and second layer
    model.add(Dense(input_dim = 387, units=params['units1'], kernel_initializer = "glorot_uniform")) 
    model.add(BatchNormalization())
    model.add(activation_func(params['activation_func']))
    model.add(Dropout(params['dropout1']))

	# Create second layer
    if params['choice']['layers']== 'two':
        model.add(Dense(units=params['choice']['units2'], kernel_initializer = "glorot_uniform")) 
        model.add(BatchNormalization())
        model.add(activation_func(params['activation_func']))
        model.add(Dropout(params['choice']['dropout2']))    

	# Create output layer
    model.add(Dense(units=1, kernel_initializer = 'glorot_uniform'))
    model.add(BatchNormalization())
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])

	# Calculate target class weight
    class_weight = compute_class_weight('balanced', np.unique(Y_train), Y_train)
    class_weight_dict = dict(enumerate(class_weight))

	# Create early stopping
    early_stopping = EarlyStopping(monitor='val_acc', patience=20, mode='auto')

	# Fit model
    model.fit(X_train, Y_train, batch_size=params['batch_size'], epochs = 1000, validation_data=(X_val, Y_val), class_weight=class_weight_dict, callbacks=[early_stopping], verbose = 0)

	# Make predictions
    Y_pred_prob = model.predict(X_test, batch_size=params['batch_size'], verbose = 0)
        
    # Calculate AUC score 
    auc_score = roc_auc_score(Y_test, Y_pred_prob)
    
    print('AUC:', auc_score)
    sys.stdout.flush() 
    return {'loss': -auc_score, 'status': STATUS_OK}

# Call hyperopt method to find model that minimises AUC  
trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=evaluations, trials=trials)
print ('Best ' + target + ' model: ')
print (best)

# Re-create model with best params
    
def run_best_model(best, X_train, Y_train, X_test, Y_test, X_val, Y_val, target):
    
    if (best.get('activation_func')) == 0: best_activation_func = 'swish'
    if (best.get('activation_func')) == 1: best_activation_func = 'PReLU'   
    if (best.get('activation_func')) == 2: best_activation_func = 'relu'

    print("Best activation function: " + best_activation_func)
    
    if (best.get('batch_size')) == 0: best_batch_size = 16
    if (best.get('batch_size')) == 1: best_batch_size = 32
    if (best.get('batch_size')) == 2: best_batch_size = 64
    if (best.get('batch_size')) == 3: best_batch_size = 128

    print("Best batch size: " + str(best_batch_size))

    best_drop_out1 = best.get('dropout1')
    print("Best drop out in layer 1: " + str(best_drop_out1))
    best_drop_out2 = best.get('dropout2')
    print("Best drop out in layer 2: " + str(best_drop_out2))

    if (best.get('num_layers')) == 0: best_num_layers = 'one'  
    if (best.get('num_layers')) == 1: best_num_layers = 'two'
        
    print("Best number of layers: " + str(best_num_layers))
    
    if (best.get('optimizer')) == 0:
        best_optimzer = 'nadam'
    
    if (best.get('optimizer')) == 1:
        best_optimzer = 'adam'
    
    if (best.get('optimizer')) == 2:
        best_optimzer = 'sgd'
      
    print("Best optimizer: " + best_optimzer)
        
    if (best.get('units1')) == 0: best_units1 = 20
    if (best.get('units1')) == 1: best_units1 = 30
    if (best.get('units1')) == 2: best_units1 = 40
    if (best.get('units1')) == 3: best_units1 = 50
    if (best.get('units1')) == 4: best_units1 = 60
    if (best.get('units1')) == 5: best_units1 = 70
    if (best.get('units1')) == 6: best_units1 = 80
    if (best.get('units1')) == 7: best_units1 = 90
    print("Best units in layer 1: " + str(best_units1))
    
    if (best_num_layers) == 'two': 
        if (best.get('units2')) == 0: best_units2 = 20
        if (best.get('units2')) == 1: best_units2 = 30
        if (best.get('units2')) == 2: best_units2 = 40
        if (best.get('units2')) == 3: best_units2 = 50
        if (best.get('units2')) == 4: best_units2 = 60
        if (best.get('units2')) == 5: best_units2 = 70
        if (best.get('units2')) == 6: best_units2 = 80
        if (best.get('units2')) == 7: best_units2 = 90
    else: best_units2 = 'None'
    print("Best units in layer 2: " + str(best_units2))
    
    # Re-run model using best params
    model = Sequential()
    model.add(Dense(input_dim = 387, units=best_units1, kernel_initializer = "glorot_uniform"))   
    if (best.get('activation_func')) == 0: model.add(Activation(swish)) 
    if (best.get('activation_func')) == 1: model.add(PReLU())   
    if (best.get('activation_func')) == 2: model.add(Activation('relu'))        
    model.add(Dropout(best_drop_out1))

    if best_num_layers== 'two':
        model.add(Dense(units=best_units2, kernel_initializer = "glorot_uniform"))
        if (best.get('activation_func')) == 0: model.add(Activation(swish))  
        if (best.get('activation_func')) == 1: model.add(PReLU())   
        if (best.get('activation_func')) == 2: model.add(Activation('relu'))
        model.add(Dropout(best_drop_out2) )   

    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=best_optimzer, metrics=['accuracy'])
    
    class_weight = compute_class_weight('balanced', np.unique(Y_train), Y_train)
    class_weight_dict = dict(enumerate(class_weight))
    
    # Create weights checkpoint for best validation accuracy
    filepath="weights.best.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='auto')

	# Create early stopping
    early_stopping = EarlyStopping(monitor='val_acc', patience=20, mode='auto')
     
    callbacks_list = [checkpoint, early_stopping]

	# Create history object to plot training history
    history = History()
    history = model.fit(X_train, Y_train, batch_size=best_batch_size, epochs = 1000, validation_data=(X_val, Y_val), class_weight=class_weight_dict, callbacks=callbacks_list, verbose = 0)

    # Plotting history

    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
    
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    # Load best weights
    model.load_weights("weights.best.hdf5")
    # Compile model (required to make predictions)
    model.compile(loss='binary_crossentropy', optimizer=best_optimzer, metrics=['accuracy'])
    
    print("Created model and loaded weights from file")

	# Make predictions using best weights
    Y_pred_prob = model.predict(X_test, batch_size = best_batch_size, verbose = 0)
     
    # Calculate AUC score 
    auc_score = roc_auc_score(Y_test, Y_pred_prob)
    print ('Re-run best ' + target + ' model AUC: ', auc_score)

	# Apply decision threshold to probabilities to calculate accuracy
    Y_pred = (Y_pred_prob > pred_threshold)
    Y_pred = Y_pred.astype(int)
    acc = accuracy_score(Y_test, Y_pred)
    
    print('Re-run best accuracy:', acc)

    # Plotting the ROC curve
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='ANN')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ANN ROC Curve')
    plt.show();
    
    return Y_pred_prob, auc_score

# Call run_best_model method
Y_pred_prob, auc_score = run_best_model(best, X_train, Y_train, X_test, Y_test, X_val, Y_val, target)

# Turning the probabilities into binary predictions for future use
Y_pred = (Y_pred_prob > pred_threshold)
Y_pred = Y_pred.astype(int)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)

print("Confusion matrx:")
print(cm)