#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 19:45:25 2020

@author: leo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 19:39:38 2020

@author: leo
"""


import tensorflow as tf


class LSTMmultiInput():
    """
    This class represents a Multi-Input LSTM model used for the task of predicting
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
    >>> from lstm_multi_input import LSTMmultiInput
    >>> lmulti = LSTMmultiInput(epochs, batch_size, loss, metric, callbacks)
    >>> lmulti.fit(X, y)
    >>> lmulti.predict(X_test)
    array([0, 1, 0.5, .........])
    """
    
    def __init__(self, epochs, batch_size, loss, metric, callbacks):
        self.num_time_steps = 30
        self.num_temporal_feats = 5
        self.num_constant_feats = 5
        self.epochs = epochs
        self.batch = batch_size
        self.loss = loss
        self.metric = metric
        self.model = None
        self.history = None
        self.es = callbacks
        
    def fit(self, X_temp, X_const, y, X_val_temp, X_val_const, y_val):      
        """        
        This method will apply the fit operation to both the constant features
        and the temporal features.
        
        The model will be trained on the training set and the early stopping 
        method will be applied on the validation set.
        
        Parameters
        ----------
        X_temp         :  array-lke
                          The temporal predictor features in the training set.
        X_const        :  array-lke
                          The constant predictor features in the training set.
        y              :  array-lke
                          The target feature in the training set. 
        X_val_temporal :  array-lke
                          The temporal predictor features in the validation set. 
        X_val_const    :  array-lke
                          The constant predictor features in the validation set.
        y_val          :  array-lke
                          The target feature in the training set. 
                
       """
    
        constant_input = tf.keras.Input(shape=(self.num_constant_feats), name="constants")        
        temporal_input = tf.keras.Input(shape=(self.num_time_steps, self.num_temporal_feats), name="temporal")
        lstm_1 = tf.keras.layers.LSTM(64, return_sequences=True)(temporal_input)
        lstm_2 = tf.keras.layers.LSTM(32, dropout=0.1, recurrent_dropout=0.1)(lstm_1)
        concat_1 = tf.keras.layers.concatenate([constant_input, lstm_2])
        dense_2 = tf.keras.layers.Dense(1)(concat_1)                
        self.model = tf.keras.Model(inputs=[temporal_input, constant_input], outputs=[dense_2])
        self.model.compile(optimizer='adam', loss=self.loss, metrics=[self.metric])
        self.history = self.model.fit({"constants": X_const, "temporal": X_temp}, y, epochs=self.epochs, callbacks=[self.es], 
                                 batch_size=self.batch, validation_data=({"constants": X_val_const, "temporal": X_val_temp}, y_val))        
        return self.history
    
    
    def predict(self, X_test_temp, X_test_const):
        """        
        This method will apply the predict operation.
        
        The trained model will be used to predict on the test set.
        
        Parameters
        ----------
        X_test_temp   :  array-lke
                         The temporal predictor features in the test set.
        X_test_const  :  array-lke
                         The constant predictor features in the test set.
        """
        return self.model.predict({"constants": X_test_const, "temporal": X_test_temp}, batch_size=self.batch)
           
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



