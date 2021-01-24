#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf

class LSTMAutoencoder():
    """
    This class represents an LSTM-Autoencoder model used for the task of predicting
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
    >>> from lstm_autoencoder import LSTMAutoencoder
    >>> lauto = LSTMAutoencoder(epochs, batch_size, loss, metric, callbacks)
    >>> lauto.fit(X, y)
    >>> lauto.predict(X_test)
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
            
        temporal_input = tf.keras.Input(shape=(self.num_time_steps, self.num_temporal_feats), name="temporal")
        lstm_1 = tf.keras.layers.LSTM(32, return_sequences=True, kernel_initializer='glorot_normal')(temporal_input)
        lstm_2 = tf.keras.layers.LSTM(16, return_sequences=True, kerbenel_initializer='glorot_normal')(lstm_1)
        lstm_3 = tf.keras.layers.LSTM(1, kernel_initializer='glorot_normal')(lstm_2)
        repeat_1 = tf.keras.layers.RepeatVector(32)(lstm_3)
        lstm_4 = tf.keras.layers.LSTM(16, return_sequences=True, kernel_initializer='glorot_normal')(repeat_1)
        lstm_5 = tf.keras.layers.LSTM(32, kernel_initializer='glorot_normal')(lstm_4) 
        dense_1 = tf.keras.layers.Dense(1)(lstm_5)        
        self.model = tf.keras.Model(inputs=temporal_input, outputs=dense_1)
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


        

