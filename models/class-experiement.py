#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 19:36:28 2020

@author: leo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 20:16:11 2020

@author: leo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from collections import defaultdict

from lstm_multi import LSTMmulti
from lstm_autoencoder import LSTMAutoencoder
from lstm_multi_input import LSTMmultiInput


# Seed to be used for al models etc.
SEED = 400

df = pd.read_csv("data/moz_13mar_df.csv")
df1 = pd.read_csv("data/moz_13mar_df1.csv")
df2 = pd.read_csv("data/moz_13mar_df2.csv")
df3 = pd.read_csv("data/moz_13mar_df3.csv")

train_df = pd.concat([df, df1])
train_df = pd.concat([train_df, df2])
train_df = pd.concat([train_df, df3])

del df1, df2, df3

test_df = pd.read_csv('data/ken_06may_df.csv')
test_df = test_df.fillna(0)


def prep_data(df, val=True, multi_input=False, classifier=False):        
    
    df = df[df['land_cover_mode_t-1'] != 17]    
    feats = [i for i in df.columns if "land_cover" not in i]    
    feat = [i for i in df[feats].columns if ("X" not in i and "Y" not in i)]
    
    df = df[feat]
    df = df.fillna(df.mean())
    
    X, y = df.drop(columns=['target']), df['target'] 
    if classifier:
        y = y.values
        y[y > 0.01] = 1
        y[y <= 0.01] = 0
        y = pd.Series(y).astype(int)
    if val:
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)
    else:
        X_train = X
        y_train = y
        
    if multi_input == False:
        scaler = MinMaxScaler(feature_range=(0,1))
        scaler.fit(X)        
        X_train = scaler.transform(X_train)
        X_train = X_train.reshape((len(X_train), 30, 8))
        y_train = y_train.values.reshape((len(y_train), 1))
        if val:
            X_val = scaler.transform(X_val)  
            X_val = X_val.reshape((len(X_val), 30, 8))
            y_val = y_val.values.reshape((len(y_val), 1))
            return X_train, y_train, X_val, y_val
        else:
            return X_train, y_train   
    else:             
        cols = [i for i in df.columns if ("X" not in i and "Y" not in i and "mode" not in i)]
        cols = [i for i in cols if ("pred" not in i and "target" not in i)]
        cols = [i for i in cols if ("dist" not in i and "slope" not in i and "elevation" not in i)]
        cols = [i for i in cols if ("soc" not in i and "slope" not in i and "elevation" not in i)]
        constant_feats = train_df[['dist_to_water_t-1', 'slope_t-1', 'elevation_t-1', 'soc_mean_t-1']]
        temporal_feats = train_df[cols]
        const_scaler = MinMaxScaler(feature_range=(0,1))
        temp_scaler = MinMaxScaler(feature_range=(0,1))
        const_scaler.fit(constant_feats)
        temp_scaler.fit(temporal_feats)
        X_train_const = X_train[constant_feats.columns]
        X_train_const = const_scaler.transform(X_train_const)
        X_train_temp = X_train[cols]
        X_train_temp = temp_scaler.transform(X_train_temp)
        X_train_temp = X_train_temp.reshape((len(X_train), 30, 4))
        if val:
            X_val_const = X_val[constant_feats.columns]
            X_val_const = const_scaler.transform(X_val_const)
            X_val_temp = X_val[cols]
            X_val_temp = temp_scaler.transform(X_val_temp)
            X_val_temp = X_val_temp.reshape((len(X_val), 30, 4))    
        
            return X_train_temp, X_train_const, y_train, X_val_temp, X_val_const, y_val
        else:
            return X_train_temp, X_train_const, y_train

    
## Prepare training data
X_train, y_train, X_val, y_val = prep_data(train_df, classifier=True)    
X_train_temp, X_train_const, y_train, X_val_temp, X_val_const, y_val = prep_data(train_df, multi_input=True, classifier=True)    
## Prepare test data
X_test, y_test = prep_data(test_df, val=False, classifier=True)
X_test_temp, X_test_const, _ = prep_data(test_df, val=False, multi_input=True, classifier=True)    


def rmse(predictions, targets):
    predictions = np.float32(predictions)
    targets = np.float32(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())


def run_experiment(model_list, folds=3):
    exp_dict = defaultdict(list)
    for model in model_list:
        if model == lmi:
            for fold in range(1, folds+1):
                print(f"*Training on fold {fold}")
                model.fit(X_train_temp, X_train_const, y_train, X_val_temp, X_val_const, y_val)
                preds = model.predict(X_test_temp, X_test_const).reshape(len(X_test,))
                preds[preds < 0.0] = 0.0
                preds[preds > 1.0] = 1.0                
                res = rmse(np.float32(preds), np.float32(y_test))
                exp_dict[model].append(np.round(res, 6))
        else:
            for fold in range(1, folds+1):
                print(f"*Training on fold {fold}")
                model.fit(X_train, y_train, X_val, y_val)
                preds = model.predict(X_test).reshape(len(X_test,))
                preds[preds < 0.0] = 0.0
                preds[preds > 1.0] = 1.0                 
                res = rmse(np.float32(preds), np.float32(y_test))
                exp_dict[model].append(np.round(res, 6))
    return exp_dict


# Hyper-parameters
EPOCHS = 1000
BATCH = 2048
LOSS = 'binary_crossentropy'
METRICS = [tf.keras.metrics.AUC(), tf.keras.metrics.Recall()]
CALLBACKS = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              restore_best_weights=True, 
                                              min_delta=0.0001, patience=10)        
    

tf.random.set_seed(SEED)

lm = LSTMmulti(EPOCHS, BATCH, LOSS, METRICS, CALLBACKS)
lmi = LSTMmultiInput(EPOCHS, BATCH, LOSS, METRICS, CALLBACKS)
la = LSTMAutoencoder(EPOCHS, BATCH, LOSS, METRICS, CALLBACKS)


lm.fit(X_train, y_train, X_val, y_val)


test_preds = lm.predict(X_test)


test_preds[test_preds < 0.0] = 0.0
test_preds[test_preds > 1.0] = 1.0

#final_preds = pd.Series(test_preds.reshape(len(test_preds),))
final_preds = pd.Series(test_preds.flatten())
test_res = rmse(np.float32(final_preds), np.float32(y_test))



test_df = pd.read_csv('data/ken_06may_df.csv')
test_df = test_df.fillna(0)


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
        assert  ind < len(_x) * len(_y)
        h = np.floor(ind / len(_x))
        w = ind - h * len(_x)
        return int(h), int(w)
    
    def pix2df(h, w):
        assert h < len(_y)
        assert w < len(_x)
        ind = h * len(_x) + w
        return int(ind)
    
    img = train_full['target'].values.reshape(len(_y), len(_x))
    print(f'img: 50:55, {img.flatten()[50:55]}')
    print(f'df: 50:55, {train_full["elevation_t-1"].values[50:55]}')
    print(f'df2img: loc 3000, {train_full["elevation_t-1"].loc[3000]} -> {img[df2pix(3000)]}')
    print(f'img2df: (34, 46), {img[34, 46]} -> {train_full["elevation_t-1"].loc[pix2df(34, 46)]}')
    return img


img_target = img_fn(test_df).reshape(181, 139)    
plt.imshow(img_target)


test_df['target'] = final_preds

img_test = img_fn(test_df).reshape(181, 139)    
plt.imshow(img_test)
