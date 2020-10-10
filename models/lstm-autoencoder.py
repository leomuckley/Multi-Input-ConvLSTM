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

from sklearn.model_selection import train_test_split

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


# clf2 = EllipticEnvelope(contamination=.20, random_state=SEED)
# clf2.fit(train_df)

# ee_scores = pd.Series(clf2.decision_function(train_df))
# clusters2 = clf2.predict(train_df)

# train_df['pred'] = clusters2
#train_df = train_df[train_df['pred'] != -1]
X, y = train_df.drop(columns=['target']), train_df['target']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)


scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(X)

X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
    
import tensorflow as tf
from tensorflow.keras.layers import Layer



callback = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', 
                                            restore_best_weights=True, min_delta=0.0001, patience=5)

####
num_data = len(X)
#num_constant_feats = 7
num_time_steps = 30
num_temporal_feats = 8

EPOCHS = 100
BATCH = 2048

#constant_input = tf.keras.Input(shape=(num_constant_feats), name="constants")

temporal_input = tf.keras.Input(shape=(num_time_steps, num_temporal_feats), name="temporal")
lstm_1 = tf.keras.layers.LSTM(32, return_sequences=True)(temporal_input)
lstm_2 = tf.keras.layers.LSTM(16, return_sequences=True)(lstm_1)
lstm_3 = tf.keras.layers.LSTM(1)(lstm_2)
repeat_1 = tf.keras.layers.RepeatVector(32)(lstm_3)
lstm_4 = tf.keras.layers.LSTM(16, return_sequences=True)(repeat_1)
lstm_5 = tf.keras.layers.LSTM(32)(lstm_4) 
dense_1 = tf.keras.layers.Dense(1)(lstm_5)

model = tf.keras.Model(inputs=temporal_input, outputs=dense_1)

model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=[tf.keras.metrics.RootMeanSquaredError()])


def rmse(predictions, targets):
    predictions = np.float32(predictions)
    targets = np.float32(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())


X_train = X_train.reshape((len(X_train), 30, 8))
X_val = X_val.reshape((len(X_val), 30, 8))
model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val),
             batch_size=BATCH, callbacks=[callback])
#model.fit(X_train, y_log_train, epochs=5)
preds = model.predict(X_val, batch_size=BATCH).reshape(len(X_val,))
res = rmse(np.float32(preds), np.float32(y_val))


##### Test model

# yy = pd.qcut(y, 10, labels=False, duplicates='drop')


# from sklearn.model_selection import StratifiedKFold
# skf = StratifiedKFold(n_splits=5, random_state=SEED)
# skf.get_n_splits(X, yy)

# X = pd.DataFrame(X)
# #print(skf)
# count = 1
# rmse_list = []
# pred_list = []
# for train_index, val_index in skf.split(X, yy):
#     print("TRAIN:", train_index, "TEST:", val_index)
#     X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
#     y_train, y_val = y.iloc[train_index], y.iloc[val_index]
#     #y_log_train = np.log1p(y_train)
    
#     X_train = X_train.values.reshape((len(X_train), 30, 8))
#     X_val = X_val.values.reshape((len(X_val), 30, 8))
#     model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val),
#                  callbacks=[callback])
#     #model.fit(X_train, y_log_train, epochs=5)
#     preds = model.predict(X_val).reshape(len(X_val,))
#     res = rmse(np.float32(preds), np.float32(y_val))
#     pred_list.append(preds)
#     fig, (ax1, ax2) = plt.subplots(1, 2)
#     fig.suptitle(f'RMSE is {res} on Fold {count}')
#     ax1.hist(preds, bins=100)
#     ax1.set_title("Predicted values")
#     ax2.hist(y_val, bins=100)
#     ax2.set_title("True values")
#     plt.show()
#     rmse_list.append(res)
#     # print(f"The RMSE on fold {count} is {res}")
#     count += 1
    
# print(f"The average RMSE is {np.mean(rmse_list)}")

# The average RMSE is 0.14830604195594788 (with 1 node)
# The average RMSE is 0.13499458134174347 (with 4 nodes)
# The average RMSE is 0.12377201020717621 (with 64, 32)
# The average RMSE is 0.12217734754085541 (huber + validation)
# The average RMSE is 0.11980126798152924 (+ slope)
# The average RMSE is 0.10974271595478058 (without outliers 20%)
# The average RMSE is 0.11043436825275421 (without land_cover)
# The average RMSE is 0.10685320198535919 (with non-neg weight constraint)

# New best 0.0971730
# The average RMSE is 0.091341

test_df = pd.read_csv('data/mwi_13mar_df.csv')
#test_df = pd.read_csv('data/mwi_13mar_df.csv')
#feats = [i for i in test_df.columns if "land_cover" not in i]
#test_feat = [i for i in test_df[feats].columns if ("X" not in i and "Y" not in i)]

test_df['land_cover_mode_t-1'] = test_df['land_cover_mode_t-1'].astype(int)

#test_df = test_df[test_df['land_cover_mode_t-1'] != 17]

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




y_test = test_df['target']
X_test = test_df.drop('target', axis=1)

X_test = X_test[train_df.drop(columns=['target']).columns]

X_test = scaler.fit_transform(X_test)


X_test = X_test.reshape((len(X_test), 30, 8))

test_preds = model.predict(X_test, batch_size=1)


test_preds[test_preds < 0.0] = 0.0
test_preds[test_preds > 1.0] = 1.0

#final_preds = pd.Series(test_preds.reshape(len(test_preds),))
final_preds = pd.Series(test_preds.flatten())

test_res = rmse(np.float32(final_preds), np.float32(y_test))


def plot_flood(res, rows, cols):
    
    flood_list = list(res)
    mat = np.array(flood_list).reshape(cols, rows)
    plt.imshow(mat.T) #, cmap='Blues')
    return plt.show()


def plot_flood_preds(res, rows, cols):
    
    flood_list = list(res)
    mat = np.array(flood_list).T.reshape(rows, cols)
    plt.imshow(mat) #, cmap='Blues')
    return plt.show()#


# def plot_flood(df, grid, xmin, xmax, ymin, ymax):
#     rows = grid.shape[0]
#     cols = grid.shape[1]
#     cols = len(list(np.arange(xmin, xmax, 0.01)))
#     rows = len(list(np.arange(ymin, ymax, 0.01)))
#     flood_list = df['target'].to_list()
#     mat = np.array(flood_list).reshape(cols, rows)
#     plt.imshow(mat.T, cmap='Blues')
#     return plt.show()


plot_flood(final_preds, rows=363, cols=142)

plot_flood(y_test, rows=363, cols=142)

res_df = pd.DataFrame(data={"X": test_df['X_t-1'], "Y": test_df['Y_t-1'], "true": y_test, "final_preds": final_preds}, dtype=float)




fl = test_df[final_preds > 0.25]['land_cover_mode_t-1'].astype(int)

non_fl = test_df[final_preds < 0.05]['land_cover_mode_t-1'].astype(int)

"""
Try He weight initialisation for Dense.
Try Xavier for LSTM (used for tanh)
"""
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
