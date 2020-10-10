#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 20:25:46 2020

@author: leo
"""

    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 17:09:48 2020

@author: leo
"""

import pandas as pd
import matplotlib.pyplot as plt

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

# onehot_encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
# #integer_encoded = integer_encoded.reshape(train_df["land_cover_t-1"], 1)
# onehot_encoded = onehot_encoder.fit_transform(train_df[["land_cover_mode_t-1", "land_cover_mode_t-2"]])
# print(onehot_encoded)
# # invert first example

#feats = [i for i in train_df.columns if "land_cover" not in i]

#onehot_encoder.get_feature_names(input_features=feats)


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

#feat = [i for i in train_df[feats].columns if ("X" not in i and "Y" not in i)]


#train_df = train_df[feat]
train_df = train_df.fillna(train_df.mean())


clf2 = EllipticEnvelope(contamination=.20, random_state=SEED)
clf2.fit(train_df)

ee_scores = pd.Series(clf2.decision_function(train_df))
clusters2 = clf2.predict(train_df)

train_df['pred'] = clusters2
train_df = train_df[train_df['pred'] != -1]
X, y = train_df.drop(columns=['pred','target']), train_df['target']

cols = [i for i in X.columns if '31' not in i]
X = X[cols]




import tensorflow as tf


callback = tf.keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', 
                                            restore_best_weights=True, min_delta=0.0001, patience=5)



# # define model where LSTM is also output layer
# model = tf.keras.Sequential()
# model.add(tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(30,5), kernel_constraint=tf.keras.constraints.NonNeg()))
# model.add(tf.keras.layers.LSTM(32, dropout=0.1, recurrent_dropout=0.1, kernel_constraint=tf.keras.constraints.NonNeg()))
# model.add(tf.keras.layers.Dense(1, kernel_constraint=tf.keras.constraints.NonNeg()))
# #model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
# model.compile(optimizer='adam', loss=tf.keras.losses.Huber(), metrics=[tf.keras.metrics.RootMeanSquaredError()])


# title_input = keras.Input(
#     shape=(None,), name="title"
# )  # Variable-length sequence of ints
# body_input = keras.Input(shape=(None,), name="body")  # Variable-length sequence of ints
# tags_input = keras.Input(
#     shape=(num_tags,), name="tags"
# )  # Binary vectors of size `num_tags`

# # Embed each word in the title into a 64-dimensional vector
# title_features = layers.Embedding(num_words, 64)(title_input)
# # Embed each word in the text into a 64-dimensional vector
# body_features = layers.Embedding(num_words, 64)(body_input)

# # Reduce sequence of embedded words in the title into a single 128-dimensional vector
# title_features = layers.LSTM(128)(title_features)
# # Reduce sequence of embedded words in the body into a single 32-dimensional vector
# body_features = layers.LSTM(32)(body_features)

# # Merge all available features into a single large vector via concatenation
# x = layers.concatenate([title_features, body_features, tags_input])

# # Stick a logistic regression for priority prediction on top of the features
# priority_pred = layers.Dense(1, name="priority")(x)
# # Stick a department classifier on top of the features
# department_pred = layers.Dense(num_departments, name="department")(x)

# # Instantiate an end-to-end model predicting both priority and department
# model = keras.Model(
#     inputs=[title_input, body_input, tags_input],
#     outputs=[priority_pred, department_pred],
# )

cols = [i for i in train_df.columns if ("X" not in i and "Y" not in i and "mode" not in i)]
cols = [i for i in cols if ("pred" not in i and "target" not in i)]
cols = [i for i in cols if ("dist" not in i and "slope" not in i and "elevation" not in i)]
cols = [i for i in cols if ("soc" not in i and "slope" not in i and "elevation" not in i)]

constant_feats = train_df[['X_t-1', 'Y_t-1', 'land_cover_mode_t-1', 'dist_to_water_t-1',
                           'slope_t-1', 'elevation_t-1', 'soc_mean_t-1']]
temporal_feats = train_df[cols]#.values.reshape((len(train_df), 30, 2))
target_feat = train_df['target']


const_scaler = MinMaxScaler(feature_range=(0,1))
temp_scaler = MinMaxScaler(feature_range=(0,1))


const_scaler.fit(constant_feats)
temp_scaler.fit(temporal_feats)

##

"""
Nh = Ns / (alpha * (Ni + N0))
"""
# Nh == number of hidden neurons
# Ni == number of input neurons
# N0 == number of input neurons
# Ns == number of samples in training set
# alpha == is scaling factor (2 - 10)

num_data = len(X)
num_constant_feats = 7
num_time_steps = 30
num_temporal_feats = 4

constant_input = tf.keras.Input(shape=(num_constant_feats), name="constants")

temporal_input = tf.keras.Input(shape=(num_time_steps, num_temporal_feats), name="temporal")
lstm_1 = tf.keras.layers.LSTM(64, return_sequences=True, kernel_constraint=tf.keras.constraints.NonNeg())(temporal_input)
lstm_2 = tf.keras.layers.LSTM(32, dropout=0.1, recurrent_dropout=0.1, kernel_constraint=tf.keras.constraints.NonNeg())(lstm_1)


#dense_1 = tf.keras.layers.Dense(32, kernel_constraint=tf.keras.constraints.NonNeg())(constant_input)

concat_1 = tf.keras.layers.concatenate([constant_input, lstm_2])


dense_2 = tf.keras.layers.Dense(1, kernel_constraint=tf.keras.constraints.NonNeg())(concat_1)


model = tf.keras.Model(
    inputs=[temporal_input, constant_input],
    outputs=[dense_2])

# import pydot
# tf.keras.utils.plot_model(model, "lstm.png", show_shapes=True)

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.Huber(),
    metrics=[tf.keras.metrics.RootMeanSquaredError()])


# model.fit(
#     {"constants": constant_feats, "temporal": temporal_feats},
#     target_feat,
#     epochs=2
#     #batch_size=32,
# )



def rmse(predictions, targets):
    predictions = np.float32(predictions)
    targets = np.float32(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())


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
    X_train_const = X_train[constant_feats.columns]
    X_val_const = X_val[constant_feats.columns]
    X_train_const = const_scaler.transform(X_train_const)
    X_val_const = const_scaler.transform(X_val_const)
    X_train_temp = X_train[cols]
    X_train_temp = temp_scaler.transform(X_train_temp)
    X_train_temp = X_train_temp.reshape((len(X_train), 30, 4))
    X_val_temp = X_val[cols]
    X_val_temp = temp_scaler.transform(X_val_temp)
    X_val_temp = X_val_temp.reshape((len(X_val), 30, 4))    
    #X_train = X_train.values.reshape((len(X_train), 30, 5))
    #X_val = X_val.values.reshape((len(X_val), 30, 5))
    model.fit({"constants": X_train_const, "temporal": X_train_temp},
               y_train, epochs=100, callbacks=[callback],
               validation_data=({"constants": X_val_const, "temporal": X_val_temp}, y_val))
    #model.fit(X_train, y_log_train, epochs=5)
    preds = model.predict({"constants": X_val_const, "temporal": X_val_temp}).reshape(len(X_val,))
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





test_df = pd.read_csv('data/zindi_orig_df.csv')

#y_test = test_df['target']
X_test = test_df.drop('target', axis=1)

#X_test = scaler.fit_transform(X_test)


X_test_const = X_test[constant_feats.columns]
X_test_temp = X_test[cols].values.reshape((len(X_test), 30, 2))

test_preds = model.predict({"constants": X_test_const, "temporal": X_test_temp}).reshape(len(X_test,))


test_preds[test_preds < 0.0] = 0.0
test_preds[test_preds > 1.0] = 1.0

final_preds = pd.Series(test_preds.reshape(len(test_preds),))

#test_res = rmse(np.float32(final_preds), np.float32(y_test))


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

plot_flood(final_preds, rows=363, cols=142)

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



#====================================================================

submission_df = pd.read_csv('SampleSubmission.csv')

submission_df['target_2019'] = final_preds

submission_df.to_csv('data/submissions/submission.csv', index = False)

