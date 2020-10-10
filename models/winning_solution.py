#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 20:00:45 2020

@author: leo
"""

import pandas as pd
import lightgbm as lgb
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon

df = pd.read_csv('data/zindi/Train (2).csv')

import geopandas as gpd
from gee_images import GeeImages



def coords_to_geom(df_):
    df = df_.copy()
    geom = [Point(df.X[i], df.Y[i]) for i in range(len(df))]
    #df['geometry'] = geom
    return geom



df['geometry'] = coords_to_geom(df)

test_gdf = gpd.GeoDataFrame(df)


# Add the land cover
data_filepath = "data/"
lc_file_list = ['new.tif']
land_cover = GeeImages(data_filepath, lc_file_list)
test_gdf = land_cover.add_all_images(test_gdf)

df['soil_major'] = test_gdf['new']


#df[(df['X'] == 34.40) & (df['Y'] == -16.03)]


df.columns

precip_features_2019 = []
precip_features_2015 = []
for col in df.columns:
    if '2019' in col:
        precip_features_2019.append(col)
    elif 'precip 2014' in col:
        precip_features_2015.append(col)
    elif 'precip 2015' in col:
        precip_features_2015.append(col)
train=tain = df[df.columns.difference(precip_features_2019)]
precip_features_2019.extend(['X',	'Y',	'elevation', 'LC_Type1_mode',	'Square_ID'])
test = df[precip_features_2019]
new_2015_cols = {}
for col, number in zip(precip_features_2015, range(1, len(precip_features_2015) + 1)):

    if 'precip' in col:
        new_2015_cols[col] = 'week_' + str(number) + '_precip'
new_2019_cols = {}
for col, number in zip(precip_features_2019, range(1, len(precip_features_2019) + 1)):
    if 'precip' in col:
        new_2019_cols[col] = 'week_' + str(number) + '_precip'
train.rename(columns = new_2015_cols, inplace = True)
test.rename(columns = new_2019_cols, inplace = True)
target = train.target_2015
train, test = train.align(test, join = 'inner', axis = 1)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error
X = train.drop(['Square_ID'], axis = 1)
y = target


train[train.columns[5:-1]].describe().loc[['mean']]

import numpy as np

train['week_7_precip_']=train['week_7_precip']+train['week_6_precip']
test['week_7_precip_']=test['week_7_precip']+test['week_6_precip']
train['week_8_precip_']=train['week_8_precip']+train['week_7_precip']
test['week_8_precip_']=test['week_8_precip']+test['week_7_precip']
train['week_9_precip_']=train['week_9_precip']+train['week_8_precip']
test['week_9_precip_']=test['week_9_precip']+test['week_8_precip']
train['max_2_weeks']=train[['week_7_precip_','week_8_precip_','week_9_precip_']].apply(np.max,axis=1)
test['max_2_weeks']=test[['week_7_precip_','week_8_precip_','week_9_precip_']].apply(np.max,axis=1)


X1=train[['LC_Type1_mode', 'X', 'Y', 'elevation','week_7_precip', 'week_8_precip', 'week_9_precip','max_2_weeks']]
sub1=test[['LC_Type1_mode', 'X', 'Y', 'elevation','week_7_precip', 'week_8_precip', 'week_9_precip','max_2_weeks']]
X1.columns=sub1.columns

def index(col):
    l=list(col)
    return l.index(max(l))

X1['max_index']=train[['week_6_precip', 'week_7_precip', 'week_8_precip','week_9_precip']].apply(index,axis=1)
sub1['max_index']=test[['week_6_precip', 'week_7_precip', 'week_8_precip','week_9_precip']].apply(index,axis=1)


sub1['slope_8_7']=((test['week_8_precip']/test['week_7_precip'])>1)*1
X1['slope_8_7']=((train['week_8_precip']/train['week_7_precip'])>1)*1


sub1['slope_9_8']=((test['week_9_precip']/test['week_8_precip'])>1)*1
X1['slope_9_8']=((train['week_9_precip']/train['week_8_precip'])>1)*1


from gee_images import GeeImages

X1['soil_major'] = df['soil_major']

from sklearn.covariance import EllipticEnvelope
clf2 = EllipticEnvelope(contamination=.17,random_state=0)
clf2.fit(X1)
ee_scores = pd.Series(clf2.decision_function(X1))
clusters2 = clf2.predict(X1)
X1['target']=target
X1['pred']=clusters2
X1=X1[X1['pred']!=-1]
X1, y =X1.drop(columns=['pred','target']), X1['target']


# from gee_images import GeeImages


# X1['geometry'] = coords_to_geom(df)

# ge = GeeImages("data/", ['ss.tif'])

# new_df = ge.add_all_images(df)

# X1.drop(columns='geometry')

# s=pd.read_csv('ss.csv')X1.X
# s.drop(columns='_soilvarie',axis=1,inplace=True)
# X1=X1.merge(s,on=['X','Y'],how='left')
# sub1=sub1.merge(s,on=['X','Y'],how='left')

def metric(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

params = { 'learning_rate':0.07,'max_depth':8}
X=X1
X_test=sub1

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

def check(col):
    if col<0:
        return 0
    elif col>1:
        return 1
    else:
        return col

preds1[preds1 < 0.0] = 0.0
preds1[preds1 > 1.0] = 1.0

preds=preds1
preds-=0.08
submission_df = pd.DataFrame({'Square_ID': test.Square_ID, 'target_2019': preds}) 
submission_df['target_2019']=submission_df['target_2019'].apply(check)
submission_df.to_csv('lili.csv', index = False)




def plot_flood(res, rows, cols):
    
    flood_list = list(res)
    mat = np.array(flood_list).reshape(cols, rows)
    plt.imshow(mat.T) #, cmap='Blues')
    return plt.show()


plot_flood(preds, 144, 161)

#new_df = pd.read_csv("data/")


# test_zipfile = "zip:///home/leo/Desktop/ml_flood_prediction/data/floods_13-03-2020/mwi_flood.zip"

df['geometry'] = coords_to_geom(df)

test_gdf = gpd.GeoDataFrame(df)

mwi_water = "zip:///home/leo/Desktop/ml_flood_prediction/data/inland_water/MWI_wat.zip"


# # Total bounds for image
# xmin,ymin,xmax,ymax = zip(test_gdf.total_bounds)



# Total bounds for image
#xmin,ymin,xmax,ymax = 34.50, -17.128481, 35.91677138449524, -13.50, 


# # Create a grid
# mwi_orig_grid = make_grid(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
# # Plot the grid
# plot_grid(flood=test_gdf, grid=mwi_orig_grid)


# # Use new grid
# mwi_new_grid = inside_border(mwi_orig_grid, country='Malawi')
# plot_grid(flood=test_gdf, grid=mwi_new_grid)


# # Get flood dataframe with fraction of grids flooded
# mwi_orig_df = flood_frac_df(gdf=test_gdf, grid=mwi_orig_grid)
# # Plot the floods
# plot_flood(mwi_orig_df, mwi_orig_grid, xmin, xmax, ymin, ymax)



# Add distance to inland water
distance = dist_to_water(test_gdf, mwi_water)
test_gdf['dist_to_water'] = distance


# Add the total precipitation
total_precip_filepath = "data/floods_13-03-2020/mwi_feats/"
tr_file_list = [f'total_precip_t-{i}.tif' for i in range(30, 0, -1)]
total_precip = GeeImages(total_precip_filepath, tr_file_list)
test_gdf = total_precip.add_all_images(test_gdf)

# Add the elevation
data_filepath = "data/floods_13-03-2020/mwi_feats/"
elev_file_list = ['elevation.tif']
elevation = GeeImages(data_filepath, elev_file_list)
test_gdf = elevation.add_all_images(test_gdf)


# Add the land cover
data_filepath = "data/floods_13-03-2020/mwi_feats/"
lc_file_list = ['land_cover_mode.tif']
land_cover = GeeImages(data_filepath, lc_file_list)
test_gdf = land_cover.add_all_images(test_gdf)


# Add surface soil moisture
soil_filepath = "data/mwi_soil_moisture/"
ssm_file_list = [f'ssm_t-{i}.tif' for i in range(28, 0, -3)]
ssm = GeeImages(soil_filepath, ssm_file_list)
test_gdf = ssm.add_all_images(test_gdf)


# Add the land cover
data_filepath = "data/floods_13-03-2020/mwi_feats/"
slope_file_list = ['slope.tif']
slope = GeeImages(data_filepath, slope_file_list)
test_gdf = slope.add_all_images(test_gdf)


# Duplicate constant features into temporal features
test_gdf = copy_col(test_gdf, column='elevation', rep=30)
test_gdf = copy_col(test_gdf, column='slope', rep=30)
test_gdf = copy_col(test_gdf, column='land_cover_mode', rep=30)

# add coordinates from centroids
#test_gdf = return_coords(test_gdf)

# Duplicate constant features into temporal features

test_gdf = copy_col(test_gdf, column='X', rep=30)
test_gdf = copy_col(test_gdf, column='Y', rep=30)


# Duplicate temporal features
test_gdf = copy_time_col(test_gdf, column_stem='ssm', start=28, end=0, gap=2, step=-3)

test_gdf['target'] = test_gdf['target_2015']
final_cols = set(df.columns).intersection(set(test_gdf.columns))

# Order the features with repsect to timesetp
header = order_feats(test_gdf)

header = [i for i in header if "precip " not in i]

header = [i for i in header if ("-0" not in i and "--" not in i)]

# Impute NaN with mean
test_gdf = test_gdf.fillna(test_gdf.mean())


write_file(test_gdf, header=header, filename="data/zindi_orig_df.csv")






