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

df = pd.read_csv("data/moz_orig_df.csv")

df1 = pd.read_csv("data/moz_aug_df1.csv")

df2 = pd.read_csv("data/moz_aug_df2.csv")

df3 = pd.read_csv("data/moz_aug_df3.csv")


train_df = pd.concat([df, df1])
train_df = pd.concat([train_df, df2])
train_df = pd.concat([train_df, df3])

del df1, df2, df3

# onehot_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
# #integer_encoded = integer_encoded.reshape(train_df["land_cover_t-1"], 1)
# onehot_encoded = onehot_encoder.fit_transform(train_df[["land_cover_mode_t-1", "land_cover_mode_t-2"]])
# print(onehot_encoded)
# # invert first example

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


clf2 = EllipticEnvelope(contamination=.20, random_state=SEED)
clf2.fit(train_df)

ee_scores = pd.Series(clf2.decision_function(train_df))
clusters2 = clf2.predict(train_df)

train_df['pred'] = clusters2
train_df = train_df[train_df['pred'] != -1]
X, y = train_df.drop(columns=['pred','target']), train_df['target']


# y = train_df['target']
# X = train_df.drop('target', axis=1)

#feat = [i for i in X.columns if "susm" not in i]

#X = X[feat]





# def preprocess_data(X, y):
        
#     # Take log of skewed target
#     y = np.log1p(y)
    
#     # Compute Z-score to remove outliers
#     z = np.abs(stats.zscore(y))
#     #print(z)   
#     threshold = 2.5
#     idx = np.where(z < threshold)[0]
#     y = y.iloc[idx]
#     X = X.iloc[idx, :]
    
#     return X, y


# X, y = preprocess_data(X, y)



# X_train = X[:27500]
# X_test = X[27500:]


scaler = MinMaxScaler(feature_range=(0,1))

import tensorflow as tf

# def get_huber_loss_fn(**huber_loss_kwargs):
#     """ Keras wrapper for loss function. """
# def custom_huber_loss(y_true, y_pred):
#     return tf.keras.losses.Huber(y_true, y_pred)
    # return custom_huber_loss

callback = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', 
                                            restore_best_weights=True, min_delta=0.0001, patience=2)



# define model where LSTM is also output layer
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(30,4), kernel_constraint=tf.keras.constraints.NonNeg()))
model.add(tf.keras.layers.LSTM(32, dropout=0.1, recurrent_dropout=0.1, kernel_constraint=tf.keras.constraints.NonNeg()))
model.add(tf.keras.layers.Dense(1, kernel_constraint=tf.keras.constraints.NonNeg()))
#model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=[tf.keras.metrics.RootMeanSquaredError()])

# X_train = X_train.reshape((len(X_train), 30, 4))

# model.fit(X_train, y_train, epochs=5)

# X_test = scaler.fit_transform(X_test)
# X_test = X_test.reshape((len(X_test), 30, 1))



# preds = model.predict(X_test).reshape(len(X_test,))


def rmse(predictions, targets):
    predictions = np.float32(predictions)
    targets = np.float32(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())

# rmse(preds, y_test)





##### Test model

yy = pd.qcut(y, 10, labels=False, duplicates='drop')


from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, random_state=SEED)
skf.get_n_splits(X, yy)

X = pd.DataFrame(X)
#print(skf)
count = 1
rmse_list = []
pred_list = []
for train_index, val_index in skf.split(X, yy):
    print("TRAIN:", train_index, "TEST:", val_index)
    X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    #y_log_train = np.log1p(y_train)
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)
    X_train = X_train.reshape((len(X_train), 30, 4))
    X_val = X_val.reshape((len(X_val), 30, 4))
    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val),
                 callbacks=[callback])
    #model.fit(X_train, y_log_train, epochs=5)
    preds = model.predict(X_val).reshape(len(X_val,))
    res = rmse(np.float32(preds), np.float32(y_val))
    pred_list.append(preds)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(f'RMSE is {res} on Fold {count}')
    ax1.hist(preds, bins=100)
    ax1.set_title("Predicted values")
    ax2.hist(y_val, bins=100)
    ax2.set_title("True values")
    plt.show()
    rmse_list.append(res)
    # print(f"The RMSE on fold {count} is {res}")
    count += 1
    
print(f"The average RMSE is {np.mean(rmse_list)}")

# The average RMSE is 0.14830604195594788 (with 1 node)
# The average RMSE is 0.13499458134174347 (with 4 nodes)
# The average RMSE is 0.12377201020717621 (with 64, 32)
# The average RMSE is 0.12217734754085541 (huber + validation)
# The average RMSE is 0.11980126798152924 (+ slope)
# The average RMSE is 0.10974271595478058 (without outliers 20%)
# The average RMSE is 0.11043436825275421 (without land_cover)
# The average RMSE is 0.10685320198535919 (with non-neg weight constraint)

test_df = pd.read_csv('data/mwi_orig_df.csv')

test_feat = [i for i in test_df.columns if ("X" not in i and "Y" not in i)]
test_df = test_df[test_feat]
y_test = test_df['target']
X_test = test_df.drop('target', axis=1)

X_test = scaler.fit_transform(X_test)


X_test = X_test.reshape((len(X_test), 30, 5))

test_preds = model.predict(X_test)


test_preds[test_preds < 0.0] = 0.0
test_preds[test_preds > 1.0] = 1.0

final_preds = pd.Series(test_preds.reshape(len(test_preds),))

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
    return plt.show()

plot_flood_preds(final_preds, rows=363, cols=142)

plot_flood(y_test, rows=363, cols=142)

res_df = pd.DataFrame(data={"true": y_test, "final_preds": final_preds}, dtype=float)


"""
Maybe try a binary classifier?????
"""

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
