#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import LSTM, RepeatVector, Dense, Flatten, ConvLSTM2D, TimeDistributed




class ConvLSTM():
    """
    This class represents an Multivariate ConvLSTM model used for the task of predicting
    flood extent.
    
    
    Parameters
    ----------
    epochs   :  int
                This parameters determines the number passes through the data
                that that the model will train.
                 
    batch_size  :  int
                   The batch size will determine after how many samples the 
                   model update the weights.
                       
    loss    :  tf.keras.losses object
               Example: tf.keras.loss.Mean_Squared_Error
               
    metric    :  tf.keras.metrics object
                 Example: tf.keras.metrics.Root_Mean_Squared_Error
                 
    callbacks   :  tf.keras.callbacks object
                   Example: tf.keras.callbacks.EarlyStopping 
   
    
    Examples
    --------
    >>> from lstm_multi import LSTMmulti
    >>> lmulti = LSTMmulti(epochs, batch_size, loss, metric, callbacks)
    >>> lmulti.fit(X, y)
    >>> lmulti.predict(X_test)
    array([0, 1, 0.5, .........])
    
    """
    
    
    def __init__(self, epochs, batch_size, loss, metric, callbacks):
        self.num_time_steps = 30
        self.num_temporal_feats = 10
        self.epochs = epochs
        self.batch = batch_size
        self.loss = loss
        self.metric = metric
        self.model = None
        self.history = None
        self.es = callbacks
        
    def fit(self, X, y, X_val, y_val):  
        """        
        This method will apply the fit operation.
        
        The model will be trained on the training set and the early stopping 
        method will be applied on the validation set.
        
        Parameters
        ----------
        X            :  array-lke
                        The predictor features in the training set. 
                    
        y            :  array-lke
                        The target feature in the training set. 
                    
        X_val        :  array-lke
                        The predictor features in the validation set. 
                    
        y_val        :  array-lke
                        The target feature in the training set. 
                
       """
        # Convert the input data to tensors
        X_train = X.reshape((len(X), self.num_time_steps, self.num_temporal_feats))
        y_train = y.values.reshape((len(y), 1))
        X_val = X_val.reshape((len(X_val), self.num_time_steps, self.num_temporal_feats))
        y_val = y_val.values.reshape((len(y_val), 1))
        train_x = X_train.reshape((X_train.shape[0], 1, 1, self.num_time_steps, self.num_temporal_feats))
        train_y = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
        val_x = X_val.reshape((X_val.shape[0], 1, 1, self.num_time_steps, self.num_temporal_feats))
        val_y = y_val.reshape((y_val.shape[0], y_val.shape[1], 1))
        
        # ConvLstm architecture
        model = tf.keras.Sequential()
        model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), 
                             input_shape=(1, 1, self.num_time_steps, self.num_temporal_feats)))
        model.add(Flatten())
        model.add(RepeatVector(X.shape))
        model.add(LSTM(32, return_sequences=True))
        model.add(LSTM(16, dropout=0.1, recurrent_dropout=0.1))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=[tf.keras.metrics.RootMeanSquaredError()])
        # fit network
        self.history = model.fit(train_x, train_y, epochs=self.epoch, batch_size=self.batch, validation_data=(val_x, val_y), callbacks=[self.es])

        return self.history
    
    def predict(self, X_test):
        """        
        This method will apply the predict operation.
        
        The trained model will be used to predict on the test set.
        
        Parameters
        ----------
        X_test       :  array-lke
                        The predictor features in the test set.
        """
        return self.model.predict(X_test, batch_size=self.batch)
           
    def save(self, filename):
        """        
        This method will save the trained model.
        
        Parameters
        ----------
        filename       :  str
                          The name of the file to be saved.
        """
        return self.model.save(filename)
   
    def return_history(self):
        """ This method will return the history object of the trained model."""
        return self.history



