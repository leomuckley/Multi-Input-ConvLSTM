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
        return self.model.predict({"constants": X_test_const, "temporal": X_test_temp}, batch_size=self.batch)
           
    def save(self, filename):
        return self.model.save(filename)
    
    def return_history(self):
        return self.history

