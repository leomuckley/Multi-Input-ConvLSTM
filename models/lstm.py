#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt

class LSTMAutoencoder():
    
    def __init__(self, epochs, batch_size, loss, metric):
        self.num_time_steps = 30
        self.num_temporal_feats = 8
        self.epochs = epochs
        self.batch = batch_size
        self.loss = loss
        self.metric = metric
        self.model = None
        self.history = None
        self.es = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',
                                              restore_best_weights=True, min_delta=0.0001, patience=10)        
    
    def fit(self, X, y, X_val, y_val):       
        temporal_input = tf.keras.Input(shape=(self.num_time_steps, self.num_temporal_feats), name="temporal")
        lstm_1 = tf.keras.layers.LSTM(32, return_sequences=True, kernel_initializer='glorot_normal')(temporal_input)
        lstm_2 = tf.keras.layers.LSTM(16, return_sequences=True, kernel_initializer='glorot_normal')(lstm_1)
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
        return self.model.predict(X_test, batch_size=self.batch)
           
    def save(self, filename):
        return self.model.save(filename)
        

