#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt


class LSTMmulti():
    """
    This class represents an Multivariate LSTM model used for the task of predicting
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
        # define model where LSTM is also output layer
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(self.num_time_steps, self.num_temporal_feats)))
        self.model.add(tf.keras.layers.LSTM(32, dropout=0.1, recurrent_dropout=0.1))
        self.model.add(tf.keras.layers.Dense(1))
        self.model.compile(optimizer='adam', loss=self.loss, metrics=[self.metric])
        self.history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch, validation_data=(X_val, y_val), callbacks=[self.es])
        
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



