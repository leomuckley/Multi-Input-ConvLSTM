#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 17:09:48 2020

@author: leo
"""

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




# Import Libraries
import numpy as np
import pandas as pd
from scipy import stats
import lightgbm as lgb

from bayes_opt import BayesianOptimization

# Seed to be used for al models etc.
SEED = 400

df1 = pd.read_csv("moz_orig_df.csv")

df2 = pd.read_csv("moz_aug_df.csv")

df3 = pd.read_csv("moz_diam_df.csv")


train_df = pd.concat([df1, df2])
train_df = pd.concat([train_df, df3])


train_df = train_df[train_df['target'] > 0.0]

y = train_df['target']
X = train_df.drop('target', axis=1)



def preprocess_data(X, y):
        
    # Take log of skewed target
    y = np.log1p(y)
    
    # Compute Z-score to remove outliers
    z = np.abs(stats.zscore(y))
    #print(z)   
    threshold = 2.5
    idx = np.where(z < threshold)[0]
    y = y.iloc[idx]
    X = X.iloc[idx, :]
    
    return X, y


X, y = preprocess_data(X, y)



X_train = X[:8000]
X_test = X[8000:]


# X = data.iloc[:, :16]
# y = data['precip_wk-1']
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# You normalize the last bit of remaining data
scaler.fit(X)
X = scaler.transform(X)

y_train = y[:8000]
y_test = y[8000:]

# Example of one output for whole sequence
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np

# define model where LSTM is also output layer
model = Sequential()
model.add(LSTM(1, return_sequences=True, input_shape=(30,1)))
model.add(LSTM(1, dropout=0.1, recurrent_dropout=0.1))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

X_train = X_train.values.reshape((len(X_train), 30, 1))

model.fit(X_train, y_train, epochs=5)

X_test = X_test.values.reshape((len(X_test), 30, 1))


preds = model.predict(X_test).reshape(len(X_test,))


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rmse(preds, y_test)





##### Test model

yy = pd.qcut(y, 10, labels=False, duplicates='drop')


from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5)
skf.get_n_splits(X, yy)

X = pd.DataFrame(X)
#print(skf)
count = 1
for train_index, test_index in skf.split(X, yy):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    X_train = X_train.values.reshape((len(X_train), 30, 1))
    model.fit(X_train, y_train, epochs=5)
    X_test = X_test.values.reshape((len(X_test), 30, 1))
    preds = model.predict(X_test).reshape(len(X_test,))
    print(f"The RMSE on fold {count} is {rmse(preds, y_test)}")




    





