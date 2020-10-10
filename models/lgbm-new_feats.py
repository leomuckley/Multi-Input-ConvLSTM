#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 21:48:47 2020

@author: leo
"""

import pandas as pd
import lightgbm as lgb
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from sklearn.covariance import EllipticEnvelope

import geopandas as gpd
from gee_images import GeeImages

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error

# Seed to be used for al models etc.
SEED = 400

#df = pd.read_csv("data/moz_orig_df.csv")

#df1 = pd.read_csv("data/moz_aug_df1.csv")

#df2 = pd.read_csv("data/moz_aug_df2.csv")

#df3 = pd.read_csv("data/moz_aug_df3.csv")

df = pd.read_csv("data/moz_13mar_df.csv")

df1 = pd.read_csv("data/moz_13mar_df1.csv")

df2 = pd.read_csv("data/moz_13mar_df2.csv")

df3 = pd.read_csv("data/moz_13mar_df3.csv")

train_df = pd.concat([df, df1])
train_df = pd.concat([train_df, df2])
train_df = pd.concat([train_df, df3])

del df1, df2, df3

# zindi_df = pd.read_csv("data/zindi/Train.csv")


# def split_sets(df_):
#     df = df_.copy()
#     precip_features_2019 = []
#     precip_features_2015 = []
#     for col in df.columns:
#         if '2019' in col:
#             precip_features_2019.append(col)
#         elif 'precip 2014' in col:
#             precip_features_2015.append(col)
#         elif 'precip 2015' in col:
#             precip_features_2015.append(col)
#     train = df[df.columns.difference(precip_features_2019)]
#     precip_features_2019.extend(['X',	'Y',	'elevation', 'LC_Type1_mode',	'Square_ID'])
#     test = df[precip_features_2019]
#     new_2015_cols = {}
#     for col, number in zip(precip_features_2015, range(1, len(precip_features_2015) + 1)):
    
#         if 'precip' in col:
#             new_2015_cols[col] = 'week_' + str(number) + '_precip'
#     new_2019_cols = {}
#     for col, number in zip(precip_features_2019, range(1, len(precip_features_2019) + 1)):
#         if 'precip' in col:
#             new_2019_cols[col] = 'week_' + str(number) + '_precip'
#     train.rename(columns = new_2015_cols, inplace = True)
#     test.rename(columns = new_2019_cols, inplace = True)
#     #target = train.target_2015
#     train, test = train.align(test, join = 'inner', axis = 1)
    
#     return train, test

# _, test = split_sets(zindi_df)




# def convert_to_zindi(df_train, df_test):
#     train = df_train.copy()
#     test = df_test.copy()
    
#     train['week_9_precip'] = train[train[[i for i in train.columns if 'max_precip' in i]].columns[-7:]].max().max()
#     train['week_8_precip'] = train[train[[i for i in train.columns if 'max_precip' in i]].columns[-14:-7]].max().max()
#     train['week_7_precip'] = train[train[[i for i in train.columns if 'max_precip' in i]].columns[-21:-14]].max().max()
#     train['week_6_precip'] = train[train[[i for i in train.columns if 'max_precip' in i]].columns[-14:-21]].max().max()
    
#     train['week_7_precip_']=train['week_7_precip']+train['week_6_precip']
#     test['week_7_precip_']=test['week_7_precip']+test['week_6_precip']
#     train['week_8_precip_']=train['week_8_precip']+train['week_7_precip']
#     test['week_8_precip_']=test['week_8_precip']+test['week_7_precip']
#     train['week_9_precip_']=train['week_9_precip']+train['week_8_precip']
#     test['week_9_precip_']=test['week_9_precip']+test['week_8_precip']
#     train['max_2_weeks']=train[['week_7_precip_','week_8_precip_','week_9_precip_']].apply(np.max,axis=1)
#     test['max_2_weeks']=test[['week_7_precip_','week_8_precip_','week_9_precip_']].apply(np.max,axis=1)
    
#     train['LC_Type1_mode'] = train['land_cover_mode_t-1'].astype(int)
#     #test['LC_Type1_mode'] = test['land_cover_mode_t-1']
    
#     train['X'] = train['X_t-1']
#     train['Y'] = train['Y_t-1']
    
#     train['elevation'] = train['elevation_t-1']
#     #test['elevation'] = test['elevation_t-1']
    
#     def coords_to_geom(df_):
#         df = df_.copy()
#         geom = [Point(df.X[i], df.Y[i]) for i in range(len(df))]
#         #df['geometry'] = geom
#         return geom
        
#     test['geometry'] = coords_to_geom(test)
        
#     # Add the land cover
#     data_filepath = "data/"
#     lc_file_list = ['mwi_soil_organic_carbon-4326.tif']
#     test_gdf = gpd.GeoDataFrame(test)
#     land_cover = GeeImages(data_filepath, lc_file_list)
#     test_gdf = land_cover.add_all_images(test_gdf)
    
#     test['soil_major'] = test_gdf['mwi_soil_organic_carbon-4326'].astype(int)

#     def index(col):
#         l=list(col)
#         return l.index(max(l))
    
#     train['max_index'] = train[['week_6_precip', 'week_7_precip', 'week_8_precip','week_9_precip']].apply(index,axis=1)
#     test['max_index'] = test[['week_6_precip', 'week_7_precip', 'week_8_precip','week_9_precip']].apply(index,axis=1)
    
#     train['slope_8_7'] = ((train['week_8_precip']/train['week_7_precip'])>1)*1   
#     test['slope_8_7'] = ((test['week_8_precip']/test['week_7_precip'])>1)*1
            
#     train['slope_9_8'] = ((train['week_9_precip']/train['week_8_precip'])>1)*1
#     test['slope_9_8'] = ((test['week_9_precip']/test['week_8_precip'])>1)*1
        
#     train['soil_major'] = train['soc_mean_t-1'].astype(int)
    
#     #test['soil_major'] = test['soc_mean_t-1']
    
#     cols_to_use = ['LC_Type1_mode', 'X', 'Y', 'elevation','week_7_precip', 'week_8_precip', 'week_9_precip',
#                    'max_2_weeks', 'slope_8_7', 'slope_9_8', 'soil_major']
    
#     return train[cols_to_use], test[cols_to_use]

# train, test = convert_to_zindi(df, test)

# X1 = test

# clf2 = EllipticEnvelope(contamination=.17,random_state=0)
# clf2.fit(X1)
# ee_scores = pd.Series(clf2.decision_function(X1))
# clusters2 = clf2.predict(X1)
# X1['target'] = df['target']
# X1['pred']=clusters2
# X1 = X1[X1['pred']!=-1]
# X1, y = X1.drop(columns=['pred','target']), X1['target']

#feats = [i for i in X.columns if "precip" in i]

test_df = pd.read_csv("data/zindi_orig_df.csv")

X, y= train_df.drop(columns='target'), train_df['target']


y_test = test_df['target']
X_test = test_df.drop('target', axis=1)

X_test = X_test[train_df.drop(columns=['target']).columns]

X_test = scaler.fit_transform(X_test)


#X_test = X_test.reshape((len(X_test), 30, 8))

t#est_preds = model.predict(X_test)




def metric(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

params = { 'learning_rate':0.07,'max_depth':8, 'num_leaves':2*8}
# X=X1
# X_test=test

n_estimators = 221

n_iters = 5
preds_buf = []
err_buf = []
for i in range(n_iters): 
    x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=i)
    d_train = lgb.Dataset(x_train, label=y_train)
    d_valid = lgb.Dataset(x_valid, label=y_valid)
    watchlist = [d_valid]

    model = lgb.train(params, d_train, n_estimators, watchlist, verbose_eval=1)

    preds = model.predict(x_valid)
   
 
    err_buf.append(metric(model.predict(x_valid),y_valid))
    
    
    preds = model.predict(X_test)
    
    preds_buf.append(preds)

print('Mean RMSLE = ' + str(np.mean(err_buf)) + ' +/- ' + str(np.std(err_buf)))
# Average predictions
preds1 = np.mean(preds_buf, axis=0)


test_preds[test_preds < 0.0] = 0.0
test_preds[test_preds > 1.0] = 1.0

final_preds = pd.Series(test_preds.reshape(len(test_preds),))

test_res = rmse(np.float32(final_preds), np.float32(y_test))


def check(col):
    if col<0:
        return 0
    elif col>1:
        return 1
    else:
        return col

preds=preds1
preds-=0.08
submission_df = pd.DataFrame({'Square_ID': zindi_df.Square_ID, 'target_2019': preds}) 
submission_df['target_2019']=submission_df['target_2019'].apply(check)
submission_df.to_csv('lili.csv', index = False)



