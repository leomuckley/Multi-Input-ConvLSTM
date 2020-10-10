#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 18:03:53 2020

@author: leo
"""

import pandas as pd

ken = pd.read_csv("data/ken_06may_df.csv")

mwi = pd.read_csv("data/mwi_13mar_df.csv")

moz = pd.read_csv("data/moz_13mar_df.csv")

df['land_cover_mode_t-1'] = df['land_cover_mode_t-1'].astype(int)

lc_vals = df['land_cover_mode_t-1'].unique()

df[df['land_cover_mode_t-1'] == 8]['target'].hist(bins=100)

X = df[df['land_cover_mode_t-1'] == 11]


data = [pd.Series(np.log1p(moz.target.values), name='Mozambique'),
        pd.Series(np.log1p(mwi.target.values), name='Malawi'),
        pd.Series(np.log1p(ken.target.values), name='Kenya')]


sns.set(style="whitegrid")

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]

ax = sns.violinplot(data=data)
ax.set_ylabel('')    
# ax.set_xlabel('')
ax.set(xlabel=data[1].name)#, ylabel='common ylabel')

plt.show()
sns.violinplot(data=data[1], color=flatui[1])
sns.violinplot(data=data[2], color=flatui[2])




data = [pd.Series(moz.target.values, name='Mozambique'),
        pd.Series(mwi.target.values, name='Malawi'),
        pd.Series(ken.target.values, name='Kenya')]



import pysal as ps
import pandas as pd
import numpy as np
#from pysal.contrib.viz import mapping as maps

import esda
import libpysal as lps
import geopandas as gpd

from shapely.geometry import Polygon, Point


ken['geometry'] = [Point(ken['X_t-1'][i], ken['Y_t-1'][i]) for i in range(len(ken))]
ken_gdf = gpd.GeoDataFrame(ken[['target', 'geometry']])



# data = ps.pdio.read_files("data/ken_06may_df.csv")

wq =  lps.weights.Queen.from_dataframe(ken_gdf)
wq.transform = 'r'

ken_lag = lps.weights.lag_spatial(wq, ken_gdf['target'])


flood = ken_gdf['target']
b, a = np.polyfit(flood, ken_lag, 1)
f, ax = plt.subplots(1, figsize=(9, 9))

plt.plot(flood, ken_lag, '.', color='firebrick')

 # dashed vert at mean of the price
plt.vlines(flood.mean(), ken_lag.min(), ken_lag.max(), linestyle='--')
 # dashed horizontal at mean of lagged price 
plt.hlines(ken_lag.mean(), flood.min(), flood.max(), linestyle='--')
ken_I = esda.moran.Moran(flood.values, wq)

# red line of best fit using global I as slope
plt.plot(flood, a + b*flood, 'r')
plt.title(f'Moran I = {np.round(ken_I.I, 2)} (P-value = {ken_I.p_sim})')
plt.ylabel('Spatial Lag of Flood')
plt.xlabel('Target')
plt.show()


mwi['geometry'] = [Point(mwi['X_t-1'][i], mwi['Y_t-1'][i]) for i in range(len(mwi))]
mwi_gdf = gpd.GeoDataFrame(mwi[['target', 'geometry']])

# data = ps.pdio.read_files("data/mwi_06may_df.csv")

wq =  lps.weights.Queen.from_dataframe(mwi_gdf)
wq.transform = 'r'

mwi_lag = lps.weights.lag_spatial(wq, mwi_gdf['target'])


flood = mwi_gdf['target']
b, a = np.polyfit(flood, mwi_lag, 1)
f, ax = plt.subplots(1, figsize=(9, 9))

plt.plot(flood, mwi_lag, '.', color='darkgoldenrod')

 # dashed vert at mean of the price
plt.vlines(flood.mean(), mwi_lag.min(), mwi_lag.max(), linestyle='--')
 # dashed horizontal at mean of lagged price 
plt.hlines(mwi_lag.mean(), flood.min(), flood.max(), linestyle='--')

# red line of best fit using global I as slope
plt.plot(flood, a + b*flood, 'darkgoldenrod')
plt.title(f'Moran I = {np.round(mwi_I.I, 2)} (P-value = {mwi_I.p_sim})')
plt.ylabel('Spatial Lag of Flood')
plt.xlabel('Target')
plt.show()

mwi_I = esda.moran.Moran(flood.values, wq)

df1 = pd.read_csv("data/moz_13mar_dropna_df1.csv")
df2 = pd.read_csv("data/moz_13mar_dropna_df2.csv")
df3 = pd.read_csv("data/moz_13mar_dropna_df3.csv")

moz = pd.concat([df, df1, df2, df3])

moz['geometry'] = [Point(moz['X_t-1'][i], moz['Y_t-1'][i]) for i in range(len(moz))]
moz_gdf = gpd.GeoDataFrame(moz[['target', 'geometry']])



# data = ps.pdio.read_files("data/moz_06may_df.csv")

wq =  lps.weights.Queen.from_dataframe(moz_gdf)
wq.transform = 'r'

moz_lag = lps.weights.lag_spatial(wq, moz_gdf['target'])


flood = moz_gdf['target']
b, a = np.polyfit(flood, moz_lag, 1)
f, ax = plt.subplots(1, figsize=(9, 9))

plt.plot(flood, moz_lag, '.', color='mediumblue')

 # dashed vert at mean of the price
plt.vlines(flood.mean(), moz_lag.min(), moz_lag.max(), linestyle='--')
 # dashed horizontal at mean of lagged price 
plt.hlines(moz_lag.mean(), flood.min(), flood.max(), linestyle='--')

# red line of best fit using global I as slope
plt.plot(flood, a + b*flood, 'b')
plt.title(f'Moran I = {np.round(moz_I.I, 2)} (P-value = {moz_I.p_sim})')
plt.ylabel('Spatial Lag of Flood')
plt.xlabel('Target')

moz_I = esda.moran.Moran(flood.values, wq)


plt.show()



mwi_north = mwi.iloc[:35500, :]

mwi_south = mwi.iloc[35500:, :]



mwi_south['geometry'] = [Point(mwi_south['X_t-1'][i], mwi_south['Y_t-1'][i]) for i in range(len(mwi_south))]
mwi_south_gdf = gpd.GeoDataFrame(mwi_south[['target', 'geometry']])

# data = ps.pdio.read_files("data/mwi_south_06may_df.csv")

wq =  lps.weights.Queen.from_dataframe(mwi_south_gdf)
wq.transform = 'r'

mwi_south_lag = lps.weights.lag_spatial(wq, mwi_south_gdf['target'])


flood = mwi_south_gdf['target']
b, a = np.polyfit(flood, mwi_south_lag, 1)
f, ax = plt.subplots(1, figsize=(9, 9))

plt.plot(flood, mwi_south_lag, '.', color='darkgoldenrod')

 # dashed vert at mean of the price
plt.vlines(flood.mean(), mwi_south_lag.min(), mwi_south_lag.max(), linestyle='--')
 # dashed horizontal at mean of lagged price 
plt.hlines(mwi_south_lag.mean(), flood.min(), flood.max(), linestyle='--')
mwi_south_I = esda.moran.Moran(flood.values, wq)

# red line of best fit using global I as slope
plt.plot(flood, a + b*flood, 'darkgoldenrod')
plt.title(f'Moran I = {np.round(mwi_south_I.I, 2)} (P-value = {mwi_south_I.p_sim})')
plt.ylabel('Spatial Lag of Flood')
plt.xlabel('Target')
plt.show()









mwi_north['geometry'] = [Point(mwi_north['X_t-1'][i], mwi_north['Y_t-1'][i]) for i in range(len(mwi_north))]
mwi_north_gdf = gpd.GeoDataFrame(mwi_north[['target', 'geometry']])

# data = ps.pdio.read_files("data/mwi_north_06may_df.csv")

wq =  lps.weights.Queen.from_dataframe(mwi_north_gdf)
wq.transform = 'r'

mwi_north_lag = lps.weights.lag_spatial(wq, mwi_north_gdf['target'])


flood = mwi_north_gdf['target']
b, a = np.polyfit(flood, mwi_north_lag, 1)
f, ax = plt.subplots(1, figsize=(9, 9))

plt.plot(flood, mwi_north_lag, '.', color='darkgoldenrod')

 # dashed vert at mean of the price
plt.vlines(flood.mean(), mwi_north_lag.min(), mwi_north_lag.max(), linestyle='--')
 # dashed horizontal at mean of lagged price 
plt.hlines(mwi_north_lag.mean(), flood.min(), flood.max(), linestyle='--')
mwi_north_I = esda.moran.Moran(flood.values, wq)

# red line of best fit using global I as slope
plt.plot(flood, a + b*flood, 'darkgoldenrod')
plt.title(f'Moran I = {np.round(mwi_north_I.I, 2)} (P-value = {mwi_north_I.p_sim})')
plt.ylabel('Spatial Lag of Flood')
plt.xlabel('Target')
plt.show()





#####





mwi_north = mwi.iloc[:16000, :]

mwi_south = mwi.iloc[16000:, :]



#mwi_south['geometry'] = [Point(mwi_south['X_t-1'][i], mwi_south['Y_t-1'][i]) for i in range(len(mwi_south))]
mwi_south_gdf = gpd.GeoDataFrame(mwi_south[['target', 'geometry']])

# data = ps.pdio.read_files("data/mwi_south_06may_df.csv")

wq =  lps.weights.Queen.from_dataframe(mwi_south_gdf)
wq.transform = 'r'

mwi_south_lag = lps.weights.lag_spatial(wq, mwi_south_gdf['target'])


flood = mwi_south_gdf['target']
b, a = np.polyfit(flood, mwi_south_lag, 1)
f, ax = plt.subplots(1, figsize=(9, 9))

plt.plot(flood, mwi_south_lag, '.', color='darkgoldenrod')

 # dashed vert at mean of the price
plt.vlines(flood.mean(), mwi_south_lag.min(), mwi_south_lag.max(), linestyle='--')
 # dashed horizontal at mean of lagged price 
plt.hlines(mwi_south_lag.mean(), flood.min(), flood.max(), linestyle='--')
mwi_south_I = esda.moran.Moran(flood.values, wq)

# red line of best fit using global I as slope
plt.plot(flood, a + b*flood, 'darkgoldenrod')
plt.title(f'Moran I = {np.round(mwi_south_I.I, 2)} (P-value = {mwi_south_I.p_sim})')
plt.ylabel('Spatial Lag of Flood')
plt.xlabel('Target')
plt.show()









#mwi_north['geometry'] = [Point(mwi_north['X_t-1'][i], mwi_north['Y_t-1'][i]) for i in range(len(mwi_north))]
mwi_north_gdf = gpd.GeoDataFrame(mwi_north[['target', 'geometry']])

# data = ps.pdio.read_files("data/mwi_north_06may_df.csv")

wq =  lps.weights.Queen.from_dataframe(mwi_north_gdf)
wq.transform = 'r'

mwi_north_lag = lps.weights.lag_spatial(wq, mwi_north_gdf['target'])


flood = mwi_north_gdf['target']
b, a = np.polyfit(flood, mwi_north_lag, 1)
f, ax = plt.subplots(1, figsize=(9, 9))

plt.plot(flood, mwi_north_lag, '.', color='darkgoldenrod')

 # dashed vert at mean of the price
plt.vlines(flood.mean(), mwi_north_lag.min(), mwi_north_lag.max(), linestyle='--')
 # dashed horizontal at mean of lagged price 
plt.hlines(mwi_north_lag.mean(), flood.min(), flood.max(), linestyle='--')
mwi_north_I = esda.moran.Moran(flood.values, wq)

# red line of best fit using global I as slope
plt.plot(flood, a + b*flood, 'darkgoldenrod')
plt.title(f'Moran I = {np.round(mwi_north_I.I, 2)} (P-value = {mwi_north_I.p_sim})')
plt.ylabel('Spatial Lag of Flood')
plt.xlabel('Target')
plt.show()


sns.set_context("paper")
sns.set_style("whitegrid", {'legend.frameon':True})

x = mwi['dist_to_water_t-1'].astype(float).values
y = ken['dist_to_water_t-1'].astype(float).values
z = moz['dist_to_water_t-1'].astype(float).values

x1 = mwi_south['land_cover_mode_t-1'].astype(int).values
x2 = mwi_north['land_cover_mode_t-1'].astype(int).values


import numpy as np; np.random.seed(10)
import seaborn as sns; sns.set(color_codes=True)
mean, cov = [0, 2], [(1, .5), (.5, 1)]
x, y = np.random.multivariate_normal(mean, cov, size=50).T
ax = sns.kdeplot(x)

ax = sns.distplot(x, color='y', label='MWI')
ax = sns.distplot(y, color='r', label='KEN')
ax = sns.distplot(z, color='b', label='MOZ')
ax.grid(False)
ax.set_xlabel('Distance to water')
ax.set_ylabel('Density')
ax.legend()


ax = sns.kdeplot(x1, bw=1.5, color='y')
ax = sns.kdeplot(x2, bw=1.5, color='r')


df = pd.read_csv("data//moz_13mar_dropna_df.csv")

df['land_cover_mode_t-1'].hist(bins=100)



df[[i for i in df.columns if "max" in i]]






