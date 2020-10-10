#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:02:55 2020

@author: leo
"""

import pandas as pd
import lightgbm as lgb
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import nearest_points

from gee_images import GeeImages

import pandas as pd
import matplotlib.pyplot as plt

# df_train = pd.read_csv('Train.csv')


# df_test = pd.read_csv('Test.csv')

# precip_cols = [f'precip_wk-{i}' for i in range(17, 0, -1)]



# rain1 = df_train[[i for i in df_train.columns if ('2014-' in i) or ('2015-' in i)]].values
# rain1 = pd.DataFrame(rain1)


# # rain2 = df_test[[i for i in df_test.columns if ('2019-' in i)]].values
# # rain2 = pd.DataFrame(rain2)

# # rain_vals = pd.concat([rain1, rain2], axis=0)

# flood_vals = df_train['target_2015']


# #data = rain_vals.rename(columns={int(i):precip_cols[i] for i in range(17)}).reset_index(drop=True)

# X = rain1[:10000]
# X_test = rain1[10000:]


# # X = data.iloc[:, :16]
# # y = data['precip_wk-1']

# y = flood_vals[:10000]
# y_test =flood_vals[10000:]

# # Example of one output for whole sequence
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout
# import numpy as np

# # define model where LSTM is also output layer
# model = Sequential()
# model.add(LSTM(1, input_shape=(17,1)))
# #model.add(LSTM(1, dropout=0.9, recurrent_dropout=0.9))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mse')

# xx = X.values.reshape((len(X), 17, 1))

# model.fit(xx, y, epochs=20)

# x_test = X_test.iloc[:, :16]
# y_test = X_test['precip_wk-1']
# xx_test = x_test.values.reshape((len(x_test), 16, 1))


# preds = model.predict(xx_test).reshape(len(preds,))


# def rmse(predictions, targets):
#     return np.sqrt(((predictions - targets) ** 2).mean())

# rmse(preds, y_test)


"""

Try using max over a three-day period.

"""


# Import Libraries
import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


# Example of one output for whole sequence
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np



from sklearn.covariance import EllipticEnvelope
from keras.utils import to_categorical


# Seed to be used for al models etc.
SEED = 400

# df = pd.read_csv("data/moz_orig_df.csv")

# df1 = pd.read_csv("data/moz_aug_df1.csv")

# df2 = pd.read_csv("data/moz_aug_df2.csv")

# df3 = pd.read_csv("data/moz_aug_df3.csv")

df = pd.read_csv("data/moz_13mar_df.csv")

df1 = pd.read_csv("data/moz_13mar_df1.csv")

df2 = pd.read_csv("data/moz_13mar_df2.csv")

df3 = pd.read_csv("data/moz_13mar_df3.csv")

train_df = pd.concat([df, df1])
train_df = pd.concat([train_df, df2])
train_df = pd.concat([train_df, df3])

del df1, df2, df3

# onehot_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
# #integer_encoded = integer_encoded.reshape(train_df["land_cover_t-1"], 1)
# onehot_encoded = onehot_encoder.fit_transform(train_df[["land_cover_mode_t-1", "land_cover_mode_t-2"]])
# print(onehot_encoded)
# # invert first example

feats = [i for i in train_df.columns if "land_cover" not in i]

# onehot_encoder.get_feature_names(input_features=feats)


# def cat_to_code(df, col):
    
#     # Define feature names
#     feats = [i for i in df.columns if col in i]
#     # Convert to integers
#     for feat in feats:
#         onehot_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
#         #print(feat)
#         cat = df[feat].astype(int).values.reshape(len(df), 1)
#         feat_idx = df.columns.get_loc(feat)
#         vec = onehot_encoder.fit_transform(cat)
#         coded_feats = onehot_encoder.get_feature_names(input_features=[feat])
#         #df = pd.DataFrame(train_X_encoded, columns=coded_feats)
        
#         df = pd.concat([df.iloc[:, :feat_idx], 
#                         pd.DataFrame(vec, columns=coded_feats, index=df.index), 
#                         df.iloc[:, feat_idx:]], axis=1)
        
#     return df
        
# train_df = cat_to_code(train_df, col="land_cover")

feat = [i for i in train_df[feats].columns if ("X" not in i and "Y" not in i)]


train_df = train_df[feat]
train_df = train_df.fillna(train_df.mean())


clf2 = EllipticEnvelope(contamination=.20, random_state=SEED)
clf2.fit(train_df)

ee_scores = pd.Series(clf2.decision_function(train_df))
clusters2 = clf2.predict(train_df)

train_df['pred'] = clusters2
train_df = train_df[train_df['pred'] != -1]
X, y = train_df.drop(columns=['pred','target']), train_df['target']

cols = [i for i in X.columns if '31' not in i]
X = X[cols]

# y = train_df['target']
# X = train_df.drop('target', axis=1)

#feat = [i for i in X.columns if "susm" not in i]

#X = X[feat]





# def preprocess_data(X, y):
        
#     # Take log of skewed target
#     y = np.log1p(y)
    
#     # Compute Z-score to remove outliers
#     z = np.abs(stats.zscore(y))
#     #print(z)   
#     threshold = 2.5
#     idx = np.where(z < threshold)[0]
#     y = y.iloc[idx]
#     X = X.iloc[idx, :]
    
#     return X, y


# X, y = preprocess_data(X, y)



# X_train = X[:27500]
# X_test = X[27500:]


scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)
    
import tensorflow as tf


# from keras.models import Model, load_model
# from keras.layers import Input, Dense, Dropout
# from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

input_dim = X.shape[1] # the # features
encoding_dim = 8 # first layer
hidden_dim = int(encoding_dim / 2) #hideen layer

nb_epoch = 30
batch_size = 128
learning_rate = 0.1

input_layer = tf.keras.layers.Input(shape=(input_dim, ))
encoder = tf.keras.layers.Dense(encoding_dim, activation="tanh", activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = tf.keras.layers.Dense(hidden_dim, activation="relu")(encoder)
decoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(encoder)
decoder = tf.keras.layers.Dense(input_dim, activation='tanh')(decoder)
autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)


autoencoder.compile()

history = autoencoder.fit(X_train, y_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    shuffle=True,
                    validation_data=(X_val, y_val),
                    verbose=1)
                    #callbacks=[checkpointer, tensorboard]).history



































