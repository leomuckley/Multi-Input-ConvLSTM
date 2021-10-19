
import tensorflow as tf
from tensorflow.keras.layers import LSTM, RepeatVector, Dense, Flatten, ConvLSTM2D, TimeDistributed


class MultiInputConvLSTM():
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
        This method will apply the fit operation.
        
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
        
        y = tf.convert_to_tensor(y.values.reshape(len(y), 1))        
        y_val = tf.convert_to_tensor(y_val.values.reshape(len(y_val), 1))
                
        X_temp = tf.reshape(X_temp, shape=(X_temp.shape[0], 1, 1, self.num_time_steps, self.num_temporal_feats))
        X_val_temp = tf.reshape(X_val_temp, shape=(X_val_temp.shape[0], 1, 1, self.num_time_steps, self.num_temporal_feats))

        constant_input = tf.keras.Input(shape=(self.num_constant_feats), name="constants")        
        temporal_input = tf.keras.Input(shape=(1, 1, self.num_time_steps, self.num_temporal_feats), name='temporal')
        
        lstm_1 = tf.keras.layers.ConvLSTM2D(filters=64, kernel_size=(1,5))(temporal_input)
        flat = tf.keras.layers.Flatten()(lstm_1)
        repeat = tf.keras.layers.RepeatVector(y.shape[1])(flat)
        lstm_2 = tf.keras.layers.LSTM(32, return_sequences=True)(repeat)
        lstm_3 = tf.keras.layers.LSTM(16, dropout=0.1, recurrent_dropout=0.1)(lstm_2)
        
        dense_1 = tf.keras.layers.Dense(16, activation='relu')(constant_input)
        flat_2 = tf.keras.layers.Flatten()(dense_1)
        #repeat_2 = tf.keras.layers.RepeatVector(y_train.shape[1])(flat_2)
        #flat_3 = tf.keras.layers.Flatten()(repeat_2)        
        concat_1 = tf.keras.layers.concatenate([flat_2, lstm_3])
        dense_2 = tf.keras.layers.Dense(1)(concat_1)                
        model = tf.keras.Model(inputs=[temporal_input, constant_input], outputs=[dense_2])
        
        model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=[tf.keras.metrics.RootMeanSquaredError()])
        self.history = model.fit({"constants": X_const, "temporal": X_temp}, y, epochs=self.epoch, callbacks=[self.es], 
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



