#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 20:24:05 2020

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

from sklearn.model_selection import train_test_split

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


## Get subset where no water bodies
train_df = train_df[train_df['land_cover_mode_t-1'] != 17]

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


flood_labels = train_df['target'] > 0.05


# clf2 = EllipticEnvelope(contamination=.20, random_state=SEED)
# clf2.fit(train_df)

# ee_scores = pd.Series(clf2.decision_function(train_df))
# clusters2 = clf2.predict(train_df)

# train_df['pred'] = clusters2
#train_df = train_df[train_df['pred'] != -1]
X, y = train_df.drop(columns=['target']), train_df['target']



# pos_features = X[flood_labels]
# neg_features = X[~flood_labels]

# pos_labels = y[flood_labels]
# neg_labels = y[~flood_labels]


# ids = np.arange(len(pos_features))
# choices = np.random.choice(ids, len(neg_features)*10)

# res_pos_features = pos_features.iloc[choices]
# res_pos_labels = pos_labels.iloc[choices]

# #res_pos_features.shape

# resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
# resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)

# order = np.arange(len(resampled_labels))
# np.random.shuffle(order)
# resampled_features = resampled_features[order]
# resampled_labels = resampled_labels[order]

#resampled_features.shape


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)


scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(X)

X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
    
import tensorflow as tf
from tensorflow.keras.layers import LSTM, RepeatVector, Dense, Flatten, ConvLSTM2D, TimeDistributed



callback = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', 
                                            restore_best_weights=True, min_delta=0.0001, patience=10)

####
# num_data = len(X)
# #num_constant_feats = 7
# num_time_steps = 30
# num_temporal_feats = 8

# EPOCHS = 10
# BATCH = 2048

# #constant_input = tf.keras.Input(shape=(num_constant_feats), name="constants")

# temporal_input = tf.keras.Input(shape=(num_time_steps, num_temporal_feats), name="temporal")
# lstm_1 = tf.keras.layers.LSTM(32, return_sequences=True, kernel_initializer='glorot_normal')(temporal_input)
# lstm_2 = tf.keras.layers.LSTM(16, return_sequences=True, kernel_initializer='glorot_normal')(lstm_1)
# lstm_3 = tf.keras.layers.LSTM(1, kernel_initializer='glorot_normal')(lstm_2)
# repeat_1 = tf.keras.layers.RepeatVector(32)(lstm_3)
# lstm_4 = tf.keras.layers.LSTM(16, return_sequences=True, kernel_initializer='glorot_normal')(repeat_1)
# lstm_5 = tf.keras.layers.LSTM(32, kernel_initializer='glorot_normal')(lstm_4) 
# dense_1 = tf.keras.layers.Dense(1)(lstm_5)

# model = tf.keras.Model(inputs=temporal_input, outputs=dense_1)
# model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=[tf.keras.metrics.RootMeanSquaredError()])

# # # reshape into [samples, time steps, rows, cols, channels]

# # ((1, 6, 64, 64, 1), (1, 64, 64, 1))

# # input_shape=(n_steps, 1, n_length, n_features)))

    
num_data = len(X)
#num_constant_feats = 7
num_time_steps = 30
num_temporal_feats = 8

EPOCHS = 1000
BATCH = 2048



X_train = X_train.reshape((len(X_train), 30, 8))
y_train = y_train.values.reshape((len(y_train), 1))
X_val = X_val.reshape((len(X_val), 30, 8))
y_val = y_val.values.reshape((len(y_val), 1))

n_steps = 1
n_length = 30
n_features = 8

n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
# reshape into subsequences [samples, time steps, rows, cols, channels]
train_x = X_train.reshape((X_train.shape[0], n_steps, 1, n_length, n_features))
# reshape output into [samples, timesteps, features]
train_y = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
# validation
val_x = X_val.reshape((X_val.shape[0], n_steps, 1, n_length, n_features))
# reshape output into [samples, timesteps, features]
val_y = y_val.reshape((y_val.shape[0], y_val.shape[1], 1))


# define model
callback = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', 
                                            restore_best_weights=True, min_delta=0.0001, patience=10)


model = tf.keras.Sequential()
model.add(ConvLSTM2D(filters=64, kernel_size=(1,3), input_shape=(n_steps, 1, n_length, n_features)))
model.add(Flatten())
model.add(RepeatVector(n_outputs))
model.add(LSTM(32, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(16, activation='relu')))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=[tf.keras.metrics.RootMeanSquaredError()])
# fit network
history = model.fit(train_x, train_y, epochs=EPOCHS, batch_size=BATCH, validation_data=(val_x, val_y), callbacks=[callback])


"""
Maybe try using a smaller test set i.e. right at the bottom of the image.

Remove water mask from images.
"""

def rmse(predictions, targets):
    predictions = np.float32(predictions)
    targets = np.float32(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())



def plot_loss(history, label):
  # Use a log scale to show the wide range of values.
  #colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

  plt.semilogy(history.epoch,  history.history['loss'],
               color='b', linestyle="--", label='Train '+label)
  plt.semilogy(history.epoch,  history.history['val_loss'], color='r', label='Val '+label)
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  
  plt.legend()


plot_loss(history, "LSTM Autoencoder")


test_df = pd.read_csv('data/mwi_13mar_df.csv')
test_df['land_cover_mode_t-1'] = test_df['land_cover_mode_t-1'].astype(int)

y_test = test_df['target'].values
X_test = test_df.drop('target', axis=1)

X_test = X_test[train_df.drop(columns=['target']).columns]

X_test = scaler.fit_transform(X_test)


X_test = X_test.reshape((len(X_test), 30, 8))
test_x = X_test.reshape((X_test.shape[0], n_steps, 1, n_length, n_features))
test_y = y_test.reshape((len(y_test), 1))

test_preds = model.predict(test_x, batch_size=1)

test_preds[test_preds < 0.0] = 0.0
test_preds[test_preds > 1.0] = 1.0

#final_preds = pd.Series(test_preds.reshape(len(test_preds),))
final_preds = pd.Series(test_preds.flatten())
y_test = pd.Series(test_y.flatten())

test_res = rmse(np.float32(final_preds), np.float32(y_test))


#=================================================================



def img_fn(df):        
    train = df.drop_duplicates(subset=['X_t-1', 'Y_t-1'], keep='first').reset_index(drop=True)
    # get all unique x, y coordinates
    _x = train['X_t-1'].round(2).unique()
    _y = train['Y_t-1'].round(2).unique()
    print(len(_x), len(_y))
    
    #and create their all possible combinations
    _xn = np.meshgrid(_x, _y)[0]
    _yn = np.meshgrid(_x, _y)[1]
    
    all_xy = np.dstack([_xn, _yn]).reshape(-1, 2)
    #with this combinations create an empty df and merge it with the original one by unique (x, y) tuples
    train_full = pd.DataFrame()
    train_full.loc[:, 'X_t-1'] = all_xy[:, 0].round(2)
    train_full.loc[:, 'Y_t-1'] = all_xy[:, 1].round(2)
    train_full.loc[:, 'id'] = train_full['X_t-1'].round(2).astype(str) + '_' + train_full['Y_t-1'].round(2).astype(str)
    id2ind = dict(zip(train_full.loc[:, 'id'].values, train_full.loc[:, 'id'].factorize()[0]))
    train_full.loc[:, 'id'] = train_full.loc[:, 'id'].map(id2ind)
    train.loc[:, 'id'] = train['X_t-1'].round(2).astype(str) + '_' + train['Y_t-1'].round(2).astype(str)
    train.loc[:, 'id'] = train.loc[:, 'id'].map(id2ind)
    del train['X_t-1'], train['Y_t-1']
    train_full = train_full.merge(train, on=['id'], how='left').sort_values(['Y_t-1', 'X_t-1'], ascending=[False, True]).reset_index(drop=True)
    #del train_full['id']
    train_full = train_full.drop_duplicates(subset=['X_t-1', 'Y_t-1'], keep='first').reset_index(drop=True)
    
    #sanity check that we can switch from IMG to DF and vice versa
    def df2pix(ind):
        assert  ind < 141 * 363
        h = np.floor(ind / 141)
        w = ind - h * 141
        return int(h), int(w)
    
    def pix2df(h, w):
        assert h < 363
        assert w < 141
        ind = h * 141 + w
        return int(ind)
    
    img = train_full['target'].values.reshape(363, 141)
    print(f'img: 50:55, {img.flatten()[50:55]}')
    print(f'df: 50:55, {train_full["elevation_t-1"].values[50:55]}')
    print(f'df2img: loc 3000, {train_full["elevation_t-1"].loc[3000]} -> {img[df2pix(3000)]}')
    print(f'img2df: (34, 46), {img[34, 46]} -> {train_full["elevation_t-1"].loc[pix2df(34, 46)]}')
    return img


img_target = img_fn(test_df).reshape(363, 141)    
plt.imshow(img_target)
plt.show()

test_df['target'] = final_preds

img_test = img_fn(test_df).reshape(363, 141)
plt.imshow(img_test)
plt.show()
