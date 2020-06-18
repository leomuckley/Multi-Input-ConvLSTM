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


# Import data set
train_df = pd.read_csv('data/training_set.csv')
test_df = pd.read_csv('data/test_set.csv')


sample_submission = pd.read_csv('data/SampleSubmission.csv')
test_ids = pd.read_csv('data/new_data.csv', usecols=['Square_ID'])

precip_cols = [f'precip_wk-{i}' for i in range(17, 0, -1)]


### Prepare data

# Features to be used in modellling
train_feats = [
        'LC_Type1_mode', 'target_2015',
        'precip_wk-17', 'precip_wk-16', 'precip_wk-15',
       'precip_wk-14', 'precip_wk-13', 'precip_wk-12', 'precip_wk-11',
       'precip_wk-10', 'precip_wk-9', 'precip_wk-8', 'precip_wk-7',
       'precip_wk-6', 'precip_wk-5', 'precip_wk-4', 'precip_wk-3',
       'precip_wk-2', 'precip_wk-1', 'elevation', 'X', 'Y', 'inland_water',
       'distance_to_road', 'distance_to_water', 'max_slope',
       'soil_ssm', 'soil_susm', 'ndvi', 'evi', 'evap_mean'
       ]


train_df = train_df[train_feats]


train_df = train_df[train_df['target_2015'] > 0.0]

y = train_df['target_2015']
X = train_df.drop('target_2015', axis=1)

# Test data set
test_df = test_df[X.columns]


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


new_X = X[precip_cols]


X = new_X[:50000]
X_test = new_X[50000:]


# X = data.iloc[:, :16]
# y = data['precip_wk-1']
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

# You normalize the last bit of remaining data
scaler.fit(X)
X = scaler.transform(X)

yy = y[:50000]
y_test = y[50000:]

# Example of one output for whole sequence
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np

# define model where LSTM is also output layer
model = Sequential()
model.add(LSTM(1, return_sequences=True, input_shape=(17,1)))
model.add(LSTM(1, dropout=0.9, recurrent_dropout=0.5))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

xx = X.reshape((len(X), 17, 1))

model.fit(xx, yy, epochs=5)

#x_test = X_test.iloc[:, :17]
xx_test = X_test.values.reshape((len(X_test), 17, 1))


preds = model.predict(xx_test).reshape(len(X_test,))


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

rmse(preds, y_test)



from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)


yy = pd.qcut(y, 2, labels=False, duplicates='drop')



from sklearn.mixture import GaussianMixture 


y_label = [1 if i < 0.25 else 0 for i in y]

#y = y.values.reshape(len(y),1)


model = GaussianMixture(n_components=2, covariance_type='full', random_state=SEED)  # 2. Instantiate the model with hyperparameters
model.fit(X)                    # 3. Fit to data. Notice y is not specified!
y_gmm = model.predict(X)



y_clust = pd.concat([pd.Series(y_label), pd.Series(y_gmm)], axis=1)
y_clust = y_clust.rename(columns={0:'Label', 1:'GMM'})


new_train_df = pd.read_csv('../data/training_set.csv')


new_train_df['clust'] = y_gmm

new_train_df[new_train_df.clust == 1]['target_2015'].hist(bins=100)

new_X = new_train_df[['elevation', 'target_2015']]


model = GaussianMixture(n_components=2, covariance_type='full', random_state=SEED)  # 2. Instantiate the model with hyperparameters
model.fit(new_X)                    # 3. Fit to data. Notice y is not specified!
y_gmm = model.predict(new_X)








