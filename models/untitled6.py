#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 10:38:59 2020

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

# for Box-Cox Transformation
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

# for min_max scaling
from mlxtend.preprocessing import minmax_scaling



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

flood_labels = (train_df['target'] > 0.01) & (train_df['target'] < 0.99)
X, y = train_df.drop(columns=['target']), train_df['target']


from shapely.geometry import Point, Polygon
from geopandas.tools import sjoin
from libpysal.weights import Kernel



test_zipfile = "zip:///home/leo/Desktop/ml_flood_prediction/data/floods_13-03-2020/mwi_flood.zip"

test_gdf = gpd.read_file(test_zipfile)


points = test_gdf.centroid
points = [tuple((i.x, i.y)) for i in points]

kw = Kernel(points[:100])
    

test_preds[test_preds < 0] = 0.0
test_preds[test_preds > 1] = 1.0




