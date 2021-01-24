#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import geopandas as gpd
import pandas as pd
import numpy as np
import json           


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from collections import defaultdict

from lstm_multi import LSTMmulti
from lstm_autoencoder import LSTMAutoencoder
from lstm_multi_input import LSTMmultiInput

from lgbm_model import LGBM

from shapely.geometry import Point, Polygon, MultiPolygon
from gee_images import GeeImages




# Seed to be used for al models etc.
SEED = 400

df = pd.read_csv("data/moz_13mar_dropna_df.csv")
df1 = pd.read_csv("data/moz_13mar_dropna_df1.csv")
df2 = pd.read_csv("data/moz_13mar_dropna_df2.csv")
df3 = pd.read_csv("data/moz_13mar_dropna_df3.csv")

train_df = pd.concat([df, df1])
train_df = pd.concat([train_df, df2])
train_df = pd.concat([train_df, df3])

train_df = train_df[train_df['land_cover_mode_t-1'] != 17]    


del df, df1, df2, df3

test_df = pd.read_csv('data/mwi_13mar_df.csv')


def prep_data(df, val=True, multi_input=False, classifier=False):        
    
    feats = [i for i in df.columns if "land_cover" not in i]    
    feat = [i for i in df[feats].columns if ("X" not in i and "Y" not in i)]
    
    df = df[feat]
    #df = df.fillna(df.mean())
    
    X, y = df.drop(columns=['target']), df['target'] 
    
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


def rmse(predictions, targets):
    predictions = np.float32(predictions)
    targets = np.float32(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())


def run_lstm_exp(model_list, folds=3, seed=SEED):
    
    tf.random.set_seed(seed)

    exp_dict = defaultdict(list)
    for model in model_list:
        if model == lmi:
            for fold in range(1, folds+1):
                print(f"*Training on fold {fold}")
                model.fit(X_train_temp, X_train_const, y_train, X_val_temp, X_val_const, y_val)
                preds = model.predict(X_test_temp, X_test_const).reshape(len(X_test,))
                preds[preds < 0.0] = 0.0
                preds[preds > 1.0] = 1.0   
                y = y_test.flatten()
                res = rmse(preds, y)
                exp_dict[model].append(np.round(res, 6))
        else:
            for fold in range(1, folds+1):
                print(f"*Training on fold {fold}")
                model.fit(X_train, y_train, X_val, y_val)
                preds = model.predict(X_test).reshape(len(X_test,))
                y = y_test.flatten()
                preds[preds < 0.0] = 0.0
                preds[preds > 1.0] = 1.0                 
                res = rmse(preds, y)
                exp_dict[model].append(np.round(res, 6))
    return exp_dict



def run_lgbm_exp(model, folds=1, seed=SEED):
    np.random.seed(SEED)
    res_list = []
    for fold in range(1, folds+1):
        print(f"*Training on fold {fold}")
        preds = model.fit_predict(seed=np.random.randint(0, 100))
        #y = y_test.flatten()
        #res_list.append(rmse(preds, y))
    return preds
    

def run_experiments(model_list):

    #results = run_lstm_exp(model_list)    
    #res_dict = {str(k):str(v) for k, v in results.items()}
        
    lgb = LGBM(train_df, test_df)
    #res_dict["lgb"] = str(run_lgbm_exp(lgb))
    p = run_lgbm_exp(lgb)
    return p

    
    
## Prepare training data
X_train, y_train, X_val, y_val = prep_data(train_df)    
X_train_temp, X_train_const, y_train, X_val_temp, X_val_const, y_val = prep_data(train_df, multi_input=True)    
## Prepare test data
X_test, y_test = prep_data(test_df, val=False)
X_test_temp, X_test_const, _ = prep_data(test_df, val=False, multi_input=True)    


test_df = test_df.sort_values(['Y_t-1', 'X_t-1'], ascending=[False, True])


lgb = LGBM(train_df, test_df)
#res_dict["lgb"] = str(run_lgbm_exp(lgb))
p = run_lgbm_exp(lgb)



def plot_flood(res, rows, cols):
    
    flood_list = list(res)
    mat = np.array(flood_list).reshape(cols, rows)
    plt.imshow(mat) #, cmap='Blues')
    return plt.show()

test_preds = p

test_preds[test_preds < 0.0] = 0.0
test_preds[test_preds > 1.0] = 1.0

final_preds = test_preds#.values.reshape(363, -1) #, order='F'))
plot_flood(final_preds, rows=-1, cols=363)

#final_preds = pd.Series(test_preds.reshape(len(test_preds),))

test_res = rmse(np.float32(final_preds), np.float32(y_test))


plot_flood(y_test, rows=-1, cols=363)


plot_flood(p, -1, 363)

plot_flood(y_test, 363, -1)







# Hyper-parameters
epoch = 1000
BATCHES = [2048, 512, 128, 32]
LOSS = tf.keras.losses.Huber()
METRIC = tf.keras.metrics.RootMeanSquaredError()
CALLBACKS = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error',
                                              restore_best_weights=True, 
                                              min_delta=0.0001, patience=10)        



result_list = []
for batch in BATCHES:
    lm = LSTMmulti(epoch, batch, LOSS, METRIC, CALLBACKS)
    lmi = LSTMmultiInput(epoch, batch, LOSS, METRIC, CALLBACKS)
    la = LSTMAutoencoder(epoch, batch, LOSS, METRIC, CALLBACKS)
    model_list = [lm, lmi, la]
    result_dict = run_experiments(model_list)
    with open(f"data/results/dropna/result-{batch}.json", "w") as outfile:  
        json.dump(result_dict, outfile)    
    result_list.append(result_dict)


