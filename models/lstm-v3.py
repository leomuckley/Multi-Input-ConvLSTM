#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 21:06:39 2020

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

df = pd.read_csv("data/moz_13mar_dropna_df.csv")

df1 = pd.read_csv("data/moz_13mar_dropna_df1.csv")

df2 = pd.read_csv("data/moz_13mar_dropna_df2.csv")

df3 = pd.read_csv("data/moz_13mar_dropna_df3.csv")

train_df = pd.concat([df, df1])
train_df = pd.concat([train_df, df2])
train_df = pd.concat([train_df, df3])

del df, df1, df2, df3


## Get subset where no water bodies
train_df = train_df[train_df['land_cover_mode_t-1'] != 17]

feats = [i for i in train_df.columns if "land_cover" not in i]

#feat = [i for i in train_df[feats].columns if ("X" not in i and "Y" not in i)]

feat = feats

train_df = train_df[feat]
train_df = train_df.fillna(train_df.mean())



def convert_coords(df):
        
    def coord_fn(x, y):
        lat = np.radians(y)
        long = np.radians(x)
        xx = np.cos(lat) * np.cos(long)
        yy = np.cos(lat) * np.sin(long)
        #zz = np.sin(lat)
        return xx, yy
    for i in range(30, 0, -1):
        df[f'X_t-{i}'], df[f'Y_t-{i}'] = coord_fn(df[f'X_t-{i}'], df[f'Y_t-{i}'])
    return df

train_df = convert_coords(train_df)


#flood_labels = (train_df['target'] > 0.01) & (train_df['target'] < 0.99)
X, y = train_df.drop(columns=['target']), train_df['target']

    
def prep_train_data(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)    
    full_cols = X.columns
    cols = feat = [i for i in X[full_cols].columns if ("X" not in i and "Y" not in i)]
    crds_cols = [i for i in X[full_cols].columns if ("X" in i or "Y" in i)]
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(X[cols])
    X_train_norm = pd.DataFrame(scaler.transform(X_train[cols]), columns=cols)
    X_val_norm = pd.DataFrame(scaler.transform(X_val[cols]), columns=cols)
    
    coord_scaler = MinMaxScaler(feature_range=(0,1))
    coord_scaler.fit(X[crds_cols])
    
    X_train_crds = pd.DataFrame(coord_scaler.transform(X_train[crds_cols]), columns=crds_cols)
    X_val_crds = pd.DataFrame(coord_scaler.transform(X_val[crds_cols]), columns=crds_cols)
        
    train_join = pd.concat([X_train_norm, X_train_crds], axis=1)
    val_join = pd.concat([X_val_norm, X_val_crds], axis=1)
    return train_join[full_cols], val_join[full_cols], y_train, y_val


X_train, X_val, y_train, y_val = prep_train_data(X, y)







import tensorflow as tf
from tensorflow.keras.layers import Layer


class LSTMmulti():
    
    def __init__(self, epochs, batch_size, loss, metric, callbacks):
        self.num_time_steps = 30
        self.num_temporal_feats = 12
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
        self.history = self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch, shuffle=False,
                                      validation_data=(X_val, y_val), callbacks=[self.es])
        
        return self.history
    
    def predict(self, X_test):
        return self.model.predict(X_test, batch_size=self.batch)
           
    def save(self, filename):
        return self.model.save(filename)
    
    def return_history(self):
        return self.history


def rmse(predictions, targets):
    predictions = np.float32(predictions)
    targets = np.float32(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())



callback = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', 
                                            restore_best_weights=True, min_delta=0.0001, patience=5)

EPOCHS = 1000
BATCH = 2048
LOSS = tf.keras.losses.Huber()
METRICS = [tf.keras.metrics.RootMeanSquaredError()]

model = LSTMmulti(EPOCHS, BATCH, LOSS, METRICS, callback)

X_train = X_train.values.reshape((len(X_train), 30, 12))

# train_tensor = tf.convert_to_tensor(X_train)

# train_tensor

# d = tf.nn.space_to_depth(input=train_tensor, block_size=4)

X_val = X_val.values.reshape((len(X_val), 30, 12))
model.fit(X_train, y_train, X_val, y_val)

# preds = model.predict(X_val, batch_size=BATCH).reshape(len(X_val,))
# res = rmse(np.float32(preds), np.float32(y_val))


## Test

test_df = pd.read_csv('data/mwi_13mar_df.csv')
test_df = test_df.fillna(0)

test_df = test_df.sort_values(['Y_t-1', 'X_t-1'], ascending=[False, True])

# test_df['centroid'] = [(test_df['X_t-1'][i], test_df['Y_t-1'][i]) for i in range(len(test_df))]

# test_df = test_df.set_index('centroid')
###

# test_df = test_df[test_df['Y_t-1'] > 1.905]

###

#test_df = test_df.iloc[:10425, :]

y_test = test_df['target']
X_test = test_df.drop('target', axis=1)

X_test = X_test[train_df.drop(columns=['target']).columns]

X_test = scaler.fit_transform(X_test)


X_test = X_test.reshape((len(X_test), 30, 12))


test_preds = model.predict(X_test)

test_preds[test_preds < 0] = 0.0
test_preds[test_preds > 1] = 1.0

import seaborn as sns
ax = sns.regplot(x=y_test, y=test_preds.flatten(), color="g")


# res = []
# for i in range(len(X_test)):
#     res.append((test_df.index[i], model.predict(X_test[i].reshape(1, 30, 8))))
    

def plot_flood(res, rows, cols):
    
    flood_list = list(res)
    mat = np.array(flood_list).reshape(cols, rows)
    plt.imshow(mat) #, cmap='Blues')
    return plt.show()


def plot_flood_preds(res, rows, cols):
    
    flood_list = list(res)
    mat = np.array(flood_list).T.reshape(rows, cols)
    plt.imshow(mat) #, cmap='Blues')
    return plt.show()#




# test_preds[test_preds < 0.0] = 0.0
# test_preds[test_preds > 1.0] = 1.0

# final_preds = test_preds.flatten().reshape(363, -1) #, order='F'))
# plot_flood(final_preds, rows=-1, cols=363)

# #final_preds = pd.Series(test_preds.reshape(len(test_preds),))

# test_res = rmse(np.float32(final_preds), np.float32(y_test))


plot_flood(y_test, rows=-1, cols=189)

# res_df = pd.DataFrame(data={"X": test_df['X_t-1'], "Y": test_df['Y_t-1'], "true": y_test, "final_preds": final_preds}, dtype=float)




# fl = test_df[final_preds > 0.25]['land_cover_mode_t-1'].astype(int)

# non_fl = test_df[final_preds < 0.05]['land_cover_mode_t-1'].astype(int)

# """
# Try He weight initialisation for Dense.
# Try Xavier for LSTM (used for tanh)
# """
#=================================================================

oof = np.zeros((363, 142))
pred_train = np.zeros(363 * 142)
pred_test = np.zeros(363 * 142)


# for j in range(n_starts):
# for i in range(n_folds):
# nn_model = torch.load(os.path.join(path_models, f'model_iter_{j}_fold_{i}.pickle'))
        
pred = model.predict(X_test)
#pred_train += pred.flatten() / n_folds / n_starts
pred_test += model.predict(X_test).flatten()         
#oof += mask * (val_folds==(i+1)) * pred

y_list = list(y_test)
img_target = np.array(y_list).reshape(142, 363)
mask = ~np.isnan(img_target)


plt.imshow(np.ma.masked_where(1-mask, pred_test.reshape(142, 363)).T)
plt.title('Test prediction')

plt.imshow(img_target.T)
plt.show()


#=================================================================



test_df = pd.read_csv('data/mwi_13mar_df.csv')
test_df = test_df.fillna(0)

test_df = test_df.sort_values(['Y_t-1', 'X_t-1'], ascending=[False, True]).reset_index(drop=True)

test_df = test_df.iloc[28400:, :]

X_test, y_test = test_df.drop(columns=['target']), test_df['target']

    
def prep_test_data(X_test, y_test):
    #X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)    
    full_cols = X.columns
    cols = feat = [i for i in X[full_cols].columns if ("X" not in i and "Y" not in i)]
    crds_cols = [i for i in X[full_cols].columns if ("X" in i or "Y" in i)]
    scaler = MinMaxScaler(feature_range=(0,1))
    scaler.fit(X[cols])
    X_test_norm = pd.DataFrame(scaler.transform(X_test[cols]), columns=cols)
       
    coord_scaler = MinMaxScaler(feature_range=(0,1))
    coord_scaler.fit(X_test[crds_cols])
    X_test_crds = pd.DataFrame(coord_scaler.transform(X_test[crds_cols]), columns=crds_cols)
    test_join = pd.concat([X_test_norm, X_test_crds], axis=1)
    
    return test_join[full_cols], y_test


X_test, y_test = prep_test_data(X_test, y_test)




prediction = np.zeros((163, 142))
row = 0 
col = 0
for coord in range(len(X_test)):
    xx, yy = X_test.iloc[coord]['X_t-1'], X_test.iloc[coord]['Y_t-1']
    print(f"{xx}, {yy}")
    prediction[row, col] += model.predict(X_test.iloc[coord].values.reshape(1, 30, 12))
    col += 1
    if col == 142:
        row += 1
        col = 0
    
prediction[prediction < 0.0] = 0.0
prediction[prediction > 1.0] = 1.0
    
plt.imshow(prediction)
plt.show()




def plot_flood(res, rows, cols):
    
    flood_list = list(res)
    mat = np.array(flood_list).reshape(cols, rows)
    plt.imshow(mat) #, cmap='Blues')
    return plt.show()

plot_flood(y_test, 142, 163)



















































#====================================================================

ids_list = []
for x, y in skf.split(X, yy):
    ids_list.append((x, y))
    

# Fold 1:
corr1 = X.iloc[ids_list[0][0]].corr()
plt.imshow(corr1)
plt.show()

# Fold 2:
corr2 = X.iloc[ids_list[1][0]].corr()
plt.imshow(corr2)
plt.show()


# Fold 3:
corr3 = X.iloc[ids_list[2][0]].corr()
plt.imshow(corr3)
plt.show()

# Fold 4:
corr4 = X.iloc[ids_list[3][0]].corr()
plt.imshow(corr4)
plt.show()

# Fold 5:
corr5 = X.iloc[ids_list[4][0]].corr()
plt.imshow(corr5)
plt.show()



#====================================================================

submission_df = pd.read_csv('data/SampleSubmission.csv')

submission_df['target_2019'] = final_preds

submission_df.to_csv('data/submissions/submission03.csv', index = False)
