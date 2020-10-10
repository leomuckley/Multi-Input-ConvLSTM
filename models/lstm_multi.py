#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 19:39:38 2020

@author: leo
"""


import tensorflow as tf
import matplotlib.pyplot as plt


class LSTMmulti():
    
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
        # define model where LSTM is also output layer
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(self.num_time_steps, self.num_temporal_feats)))
        self.model.add(tf.keras.layers.LSTM(32, dropout=0.1, recurrent_dropout=0.1))
        self.model.add(tf.keras.layers.Dense(1))
        self.model.compile(optimizer='adam', loss=self.loss, metrics=[self.metric])
        self.history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch, validation_data=(X_val, y_val), callbacks=[self.es])
        
        return self.history
    
    def predict(self, X_test):
        return self.model.predict(X_test, batch_size=self.batch)
           
    def save(self, filename):
        return self.model.save(filename)
    
    def return_history(self):
        return self.history



