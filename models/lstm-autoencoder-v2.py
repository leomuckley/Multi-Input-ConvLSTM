#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 16:41:05 2020

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



pos_features = X[flood_labels]
neg_features = X[~flood_labels]

pos_labels = y[flood_labels]
neg_labels = y[~flood_labels]


ids = np.arange(len(pos_features))
choices = np.random.choice(ids, len(neg_features)*10)

res_pos_features = pos_features.iloc[choices]
res_pos_labels = pos_labels.iloc[choices]

#res_pos_features.shape

resampled_features = np.concatenate([res_pos_features, neg_features], axis=0)
resampled_labels = np.concatenate([res_pos_labels, neg_labels], axis=0)

order = np.arange(len(resampled_labels))
np.random.shuffle(order)
resampled_features = resampled_features[order]
resampled_labels = resampled_labels[order]

#resampled_features.shape


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)


scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(X)

X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)
    
from tensorflow.keras.layers import Layer

import tensorflow as tf

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                            restore_best_weights=True, min_delta=0.0001, patience=10)


y[y > 0.01] = 1
y[y <= 0.01] = 0


####
num_data = len(X)
#num_constant_feats = 7
num_time_steps = 30
num_temporal_feats = 8

EPOCHS = 10
BATCH = 2048

#constant_input = tf.keras.Input(shape=(num_constant_feats), name="constants")

temporal_input = tf.keras.Input(shape=(num_time_steps, num_temporal_feats), name="temporal")
lstm_1 = tf.keras.layers.LSTM(32, return_sequences=True, kernel_initializer='glorot_normal')(temporal_input)
lstm_2 = tf.keras.layers.LSTM(16, return_sequences=True, kernel_initializer='glorot_normal')(lstm_1)
lstm_3 = tf.keras.layers.LSTM(1, kernel_initializer='glorot_normal')(lstm_2)
repeat_1 = tf.keras.layers.RepeatVector(32)(lstm_3)
lstm_4 = tf.keras.layers.LSTM(16, return_sequences=True, kernel_initializer='glorot_normal')(repeat_1)
lstm_5 = tf.keras.layers.LSTM(32, kernel_initializer='glorot_normal')(lstm_4) 
dense_1 = tf.keras.layers.Dense(1, activation='sigmoid')(lstm_5)

model = tf.keras.Model(inputs=temporal_input, outputs=dense_1)

#opt = tf.keras.optimizers.Adam(learning_rate=0.000001, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

#from sklearn.metrics import mean_squared_log_error

# def rmsle(y_true, y_pred):
#     rmsle = tf.math.sqrt(tf.keras.metrics.mean_squared_logarithmic_error(y_true, y_pred))
#     return rmsle


# def rmsle(y_true, y_pred, name='rmsle'):
#     def fn(y_true, y_pred):
#         rmsle = tf.math.sqrt(tf.keras.metrics.mean_squared_logarithmic_error(y_true, y_pred))
#         return rmsle
#     fn.__name__ = 'metricname_{}'.format(name)
#     return fn 
 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])


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


X_train = X_train.reshape((len(X_train), 30, 8))
y_train = y_train.reshape((len(y_train), 1))
X_val = X_val.reshape((len(X_val), 30, 8))
y_val = y_val.reshape((len(y_val), 1))
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val),
             batch_size=BATCH, callbacks=[callback])

preds = model.predict(X_val, batch_size=BATCH).reshape(len(X_val,))
res = rmse(np.float32(preds), np.float32(y_val))

plot_loss(history, "LSTM Autoencoder")

model.save('data/models/lstm_autoencoder/model-01.h5') 

m = tf.keras.models.load_model('data/models/lstm_autoencoder/model-01.h5') 

p = m.predict(X_val, batch_size=BATCH).reshape(len(X_val,))
r = rmse(np.float32(preds), np.float32(y_val))

from lstm_multi_input import LSTMmultiInput

ls = LSTMmultiInput(epochs=3, batch_size=BATCH, loss=tf.keras.losses.Huber(), metric=tf.keras.metrics.RootMeanSquaredError())

ls.fit(X_train, y_train, X_val, y_val)

ls.predict(X_val).reshape(len(X_val,))

ls.history.history['val_loss']

ls.plot(label='LSTM Autoencoder', title='Huber Loss')


ls.history.history ['val_loss']

ls.save("data/test.h5")

m = tf.keras.models.load_model('data/test.h5') 

m.predict(X_val)

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

#test_df = pd.read_csv('data/zindi_orig_df.csv')
test_df = pd.read_csv('data/mwi_13mar_df.csv')
#feats = [i for i in test_df.columns if "land_cover" not in i]
#test_feat = [i for i in test_df[feats].columns if ("X" not in i and "Y" not in i)]

test_df['land_cover_mode_t-1'] = test_df['land_cover_mode_t-1'].astype(int)

test_df['X_t-1'], test_df['X_t-1'] = test_df['X_t-1'].round(2), test_df['X_t-1'].round(2)

test_df = test_df.drop_duplicates(['X_t-1', 'Y_t-1'])

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




y_test = test_df['target'].values
X_test = test_df.drop('target', axis=1)

X_test = X_test[train_df.drop(columns=['target']).columns]

X_test = scaler.fit_transform(X_test)


X_test = X_test.reshape((len(X_test), 30, 8))
y_test = y_test.reshape((len(y_test), 1))

test_preds = model.predict(X_test[4000:6000, :], batch_size=BATCH)


test_preds[test_preds < 0.0] = 0.0
test_preds[test_preds > 1.0] = 1.0

#final_preds = pd.Series(test_preds.reshape(len(test_preds),))
#final_preds = pd.Series(test_preds.flatten())


def rmse(predictions, targets):
    predictions = np.float32(predictions)
    targets = np.float32(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())


test_res = rmse(np.float32(test_preds), np.float32(y_test))


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


plot_flood(test_preds, rows=50, cols=40)

plot_flood(y_test[4000:6000], rows=50, cols=40)

res_df = pd.DataFrame(data={"X": test_df['X_t-1'], "Y": test_df['Y_t-1'], "true": y_test, "final_preds": final_preds}, dtype=float)



fl = test_df[final_preds > 0.25]['land_cover_mode_t-1'].astype(int)

non_fl = test_df[final_preds < 0.05]['land_cover_mode_t-1'].astype(int)

"""
Try He weight initialisation for Dense.
Try Xavier for LSTM (used for tanh)
"""

#=================================================================

#test_df = pd.read_csv('data/_df.csv')


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


test_df['target'] = final_preds

img_test = img_fn(test_df).reshape(363, 141)
plt.imshow(img_test)

#=================================================================


oof = np.zeros((363, 141))
pred_train = np.zeros(363 * 141)
pred_test = np.zeros(363 * 141)


# for j in range(n_starts):
#     for i in range(n_folds):
#nn_model = torch.load(os.path.join(path_models, f'model_iter_{j}_fold_{i}.pickle'))
        
#pred = model.predict(X_test)
pred_train += pred.flatten()# / n_folds / n_starts
pred_test += model.predict(img_test).flatten() #/ n_folds / n_starts        
#oof += mask * (val_folds==(i+1)) * pred


for i in range(len(test_df)):
    print(f"True value {np.round(test_df['target'].iloc[i])}; \
          Pred value {model.predict(X_test[i, :])}")

# oof /= (j+1)
# print(f'OOF rmse: {rmse(img_target, oof, mask)}')
# print('')   

data = X_test.reshape((X_test.shape[0]*X_test.shape[1], X_test.shape[2]))

#=================================================================

oof = np.zeros((363, 141))
pred_train = np.zeros(363 * 141)
pred_test = np.zeros(363 * 141)


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
