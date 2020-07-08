#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:02:55 2020

@author: leo
"""


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon

from gee_images import GeeImages


"""
var before_start= '2019-03-01';
var before_end='2019-03-10';

// Now set the same parameters for AFTER the flood.
var after_start='2019-03-10';
var after_end='2019-03-23';

"""



def make_grid(xmin, ymin, xmax, ymax):
    
    length = 0.01
    wide = 0.01

    cols = list(np.arange(xmin, xmax, wide))
    rows = list(np.arange(ymin, ymax, length))
    rows = rows[::-1]
    
    polygons = []
    for x in cols:
        for y in rows:
            polygons.append( Polygon([(x,y), (x+wide, y), (x+wide, y-length), (x, y-length)]) )
            
    return gpd.GeoDataFrame({'geometry':polygons})


def plot_grid(flood, grid):
    ax = grid.plot(color='yellow');
    return flood.plot(ax=ax, color='black', alpha=0.5);


def flood_frac_df(gdf, grid):
    flood = gpd.sjoin(grid, gdf, how='left')
    total_list = []
    for grd in grid.geometry:
        total = 0
        for fld in gdf.geometry:
            if fld.intersects(grd):
                flood = (fld.intersection(grd)).area / grd.area
                total = total + flood
        print(total)
        total_list.append(total)
    df = pd.DataFrame(data={'target':total_list})
    df['centroid'] = grid.centroid
    df['geometry'] = grid.geometry
    return df


def plot_flood(df, grid, xmin, xmax, ymin, ymax):
    rows = grid.shape[0]
    cols = grid.shape[1]
    cols = len(list(np.arange(xmin, xmax, 0.01)))
    rows = len(list(np.arange(ymin, ymax, 0.01)))
    flood_list = df['target'].to_list()
    mat = np.array(flood_list).reshape(cols, rows)
    plt.imshow(mat.T, cmap='Blues')
    return plt.show()



def plot_flood_new(df, grid):
    rows = grid.shape[0]
    cols = grid.shape[1]
    bnds = grid.total_bounds
    xmin, ymin, xmax, ymax = bnds[0], bnds[1], bnds[2], bnds[3] 
    cols = len(list(np.arange(xmin, xmax, 0.01)))
    rows = len(list(np.arange(ymin, ymax, 0.01)))
    flood_list = df['target'].to_list()
    #zeros = list(np.zeros((rows * cols) - len(grid)))
    #full_flood = flood_list + zeros
    mat = np.array(flood_list).reshape(cols, rows)
    plt.imshow(mat.T, cmap='Blues')
    return plt.show()


def copy_col(df_, column, rep):
    df = df_.copy()
    for i in range(rep, 0, -1):
        df[f'{column}_t-{i}'] = df[column]
    df = df.drop(column, axis=1)
    return df


def copy_time_col(df_, column_stem, start, end, step, gap):
    df = df_.copy()
    for i in range(start, end-gap, step):
        for j in range(i+gap, i, -1):
            df[f'{column_stem}_t-{j}'] = df[f'{column_stem}_t-{i}']
    return df



def order_feats(df_):
    df = df_.copy()
    col = df.columns
    tup_list = [(i, np.abs(np.int(i[-2:]))) for i in col if "-" in i]
    sort_list = [b for a,b in sorted((tup[1], tup) for tup in tup_list)]
    sort_list = list(reversed([i[0] for i in sort_list]))
    target = "target"
    sort_list.append(target)
    return sort_list


def write_file(df, header, filename):
    df = df[header]    
    
    return df.to_csv(filename, header=header, index=False)


def inside_border(grid, country='Mozambique'):
    
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))    
    border = world[world.name == country]['geometry']
       
    in_border = [border.contains(i).values[0] for i in grid.geometry]
    return grid[in_border]




# df = moz_orig_df.copy()

# df['X'] = gpd.GeoSeries(df.centroid).x

# df['Y'] = gpd.GeoSeries(df.centroid).y

# df['target'] = moz_orig_df.target

# ax = pd.DataFrame(df).plot.hexbin(x='coord_x',
#                     y='coord_y',
#                     C='target',
#                     #reduce_C_function=np.sum,
#                     #gridsize=10,
#                     cmap="Blues")


################      Original Mozambique GEE Image      ######################



train_zipfile = "zip:///home/leo/Desktop/ml_flood_prediction/data/moz_orig_flood_vec.zip"

gdf = gpd.read_file(train_zipfile)


# Total bounds for image
xmin,ymin,xmax,ymax = 33.50, -20.60, 35.50, -19.00

# Create a grid
moz_orig_grid = make_grid(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
# Plot the grid
plot_grid(flood=gdf, grid=moz_orig_grid)

# moz_new_grid = inside_border(moz_orig_grid, country='Mozambique')
# plot_grid(flood=gdf, grid=moz_new_grid)


# Get flood dataframe with fraction of grids flooded
moz_orig_df = flood_frac_df(gdf=gdf, grid=moz_orig_grid)
# Plot the floods
plot_flood(moz_orig_df, moz_orig_grid, xmin, xmax, ymin, ymax)

# Add the total precipitation
total_precip_filepath = "data/moz_precip/total_precip/"
tr_file_list = [f'total_precip_t-{i}.tif' for i in range(30, 0, -1)]
total_precip = GeeImages(total_precip_filepath, tr_file_list)
moz_orig_df = total_precip.add_all_images(moz_orig_df)

# Add the elevation
data_filepath = "data/"
elev_file_list = ['elevation.tif']
elevation = GeeImages(data_filepath, elev_file_list)
moz_orig_df = elevation.add_all_images(moz_orig_df)


# Add the land cover
data_filepath = "data/"
lc_file_list = ['land_cover_mode.tif']
land_cover = GeeImages(data_filepath, lc_file_list)
moz_orig_df = land_cover.add_all_images(moz_orig_df)


# Add the slope
data_filepath = "data/"
slope_file_list = ['slope.tif']
slope = GeeImages(data_filepath, slope_file_list)
moz_orig_df = slope.add_all_images(moz_orig_df)


# Add surface soil moisture
soil_filepath = "data/soil_moisture/"
ssm_file_list = [f'ssm_t-{i}.tif' for i in range(28, 0, -3)]
ssm = GeeImages(soil_filepath, ssm_file_list)
moz_orig_df = ssm.add_all_images(moz_orig_df)



# Add the slope
data_filepath = "data/"
slope_file_list = ['slope.tif']
slope = GeeImages(data_filepath, slope_file_list)
moz_orig_df = slope.add_all_images(moz_orig_df)


# Duplicate constant features into temporal features
moz_orig_df = copy_col(moz_orig_df, column='elevation', rep=30)
moz_orig_df = copy_col(moz_orig_df, column='slope', rep=30)
moz_orig_df = copy_col(moz_orig_df, column='land_cover_mode', rep=30)


# Duplicate temporal features
moz_orig_df = copy_time_col(moz_orig_df, column_stem='ssm', start=28, end=0, gap=2, step=-3)

# Order the features with repsect to timesetp
header = order_feats(moz_orig_df)


# Impute NaN with mean
moz_orig_df = moz_orig_df.fillna(moz_orig_df.mean())


write_file(moz_orig_df, header=header, filename="data/moz_orig_df.csv")



###### Data augmentation 1: stagger x-axis by 0.5

# Total bounds for image
xmin,ymin,xmax,ymax = 33.505, -20.60, 35.495, -19.00

# Create a grid
moz_orig_grid = make_grid(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
# Plot the grid
plot_grid(flood=gdf, grid=moz_orig_grid)

# Get flood dataframe with fraction of grids flooded
moz_aug_df1 = flood_frac_df(gdf=gdf, grid=moz_orig_grid)
# Plot the floods
plot_flood(moz_aug_df1, moz_orig_grid, xmin, xmax, ymin, ymax)

# Add the total precipitation
total_precip_filepath = "data/moz_precip/total_precip/"
tr_file_list = [f'total_precip_t-{i}.tif' for i in range(30, 0, -1)]
total_precip = GeeImages(total_precip_filepath, tr_file_list)
moz_aug_df1 = total_precip.add_all_images(moz_aug_df1)

# Add the elevation
data_filepath = "data/"
elev_file_list = ['elevation.tif']
elevation = GeeImages(data_filepath, elev_file_list)
moz_aug_df1 = elevation.add_all_images(moz_aug_df1)


# Add the land cover
data_filepath = "data/"
lc_file_list = ['land_cover_mode.tif']
land_cover = GeeImages(data_filepath, lc_file_list)
moz_aug_df1 = land_cover.add_all_images(moz_aug_df1)


# Add surface soil moisture
soil_filepath = "data/soil_moisture/"
ssm_file_list = [f'ssm_t-{i}.tif' for i in range(28, 0, -3)]
ssm = GeeImages(soil_filepath, ssm_file_list)
moz_aug_df1 = ssm.add_all_images(moz_aug_df1)


# Duplicate constant features into temporal features
moz_aug_df1 = copy_col(moz_aug_df1, column='elevation', rep=30)
moz_aug_df1 = copy_col(moz_aug_df1, column='land_cover_mode', rep=30)


# Duplicate temporal features
moz_aug_df1 = copy_time_col(moz_aug_df1, column_stem='ssm', start=28, end=0, gap=2, step=-3)

# Add the slope
data_filepath = "data/"
slope_file_list = ['slope.tif']
slope = GeeImages(data_filepath, slope_file_list)
moz_aug_df1 = slope.add_all_images(moz_aug_df1)


# Duplicate constant features into temporal features
moz_aug_df1 = copy_col(moz_aug_df1, column='slope', rep=30)


# Impute NaN with mean
moz_aug_df1 = moz_aug_df1.fillna(moz_aug_df1.mean())


# Order the features with repsect to timesetp
header = order_feats(moz_aug_df1)

write_file(moz_aug_df1, header=header, filename="data/moz_aug_df1.csv")


###### Data augmentation 2: stagger y-axis by 0.5

# Total bounds for image
xmin,ymin,xmax,ymax = 33.50, -20.545, 35.50, -19.055

# Create a grid
moz_orig_grid = make_grid(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
# Plot the grid
plot_grid(flood=gdf, grid=moz_orig_grid)

# Use new grid
moz_new_grid = inside_border(moz_orig_grid, country='Mozambique')
plot_grid(flood=gdf, grid=moz_new_grid)


# Get flood dataframe with fraction of grids flooded
moz_aug_df2 = flood_frac_df(gdf=gdf, grid=moz_orig_grid)
# Plot the floods
plot_flood(moz_aug_df2, moz_orig_grid, xmin, xmax, ymin, ymax)

# Add the total precipitation
total_precip_filepath = "data/moz_precip/total_precip/"
tr_file_list = [f'total_precip_t-{i}.tif' for i in range(30, 0, -1)]
total_precip = GeeImages(total_precip_filepath, tr_file_list)
moz_aug_df2 = total_precip.add_all_images(moz_aug_df2)

# Add the elevation
data_filepath = "data/"
elev_file_list = ['elevation.tif']
elevation = GeeImages(data_filepath, elev_file_list)
moz_aug_df2 = elevation.add_all_images(moz_aug_df2)


# Add the land cover
data_filepath = "data/"
lc_file_list = ['land_cover_mode.tif']
land_cover = GeeImages(data_filepath, lc_file_list)
moz_aug_df2 = land_cover.add_all_images(moz_aug_df2)


# Add surface soil moisture
soil_filepath = "data/soil_moisture/"
ssm_file_list = [f'ssm_t-{i}.tif' for i in range(28, 0, -3)]
ssm = GeeImages(soil_filepath, ssm_file_list)
moz_aug_df2 = ssm.add_all_images(moz_aug_df2)



# Duplicate constant features into temporal features
moz_aug_df2 = copy_col(moz_aug_df2, column='elevation', rep=30)
moz_aug_df2 = copy_col(moz_aug_df2, column='land_cover_mode', rep=30)



# Duplicate temporal features
moz_aug_df2 = copy_time_col(moz_aug_df2, column_stem='ssm', start=28, end=0, gap=2, step=-3)


# Add the slope
data_filepath = "data/"
slope_file_list = ['slope.tif']
slope = GeeImages(data_filepath, slope_file_list)
moz_aug_df2 = slope.add_all_images(moz_aug_df2)


# Duplicate constant features into temporal features
moz_aug_df2 = copy_col(moz_aug_df2, column='slope', rep=30)



# Impute NaN with mean
moz_aug_df2 = moz_aug_df2.fillna(moz_aug_df2.mean())


# Order the features with repsect to timesetp
header = order_feats(moz_aug_df2)

write_file(moz_aug_df2, header=header, filename="data/moz_aug_df2.csv")


###### Data augmentation 3: stagger the x-axis by 0.5 and the y-axis by 0.5

# Total bounds for image
xmin,ymin,xmax,ymax = 33.505, -20.545, 35.495, -19.055

# Create a grid
moz_orig_grid = make_grid(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
# Plot the grid
plot_grid(flood=gdf, grid=moz_orig_grid)


# Use new grid
moz_new_grid = inside_border(moz_orig_grid, country='Mozambique')
plot_grid(flood=gdf, grid=moz_new_grid)


# Get flood dataframe with fraction of grids flooded
moz_aug_df3 = flood_frac_df(gdf=gdf, grid=moz_orig_grid)
# Plot the floods
plot_flood(moz_aug_df3, moz_orig_grid, xmin, xmax, ymin, ymax)

# Add the total precipitation
total_precip_filepath = "data/moz_precip/total_precip/"
tr_file_list = [f'total_precip_t-{i}.tif' for i in range(30, 0, -1)]
total_precip = GeeImages(total_precip_filepath, tr_file_list)
moz_aug_df3 = total_precip.add_all_images(moz_aug_df3)

# Add the elevation
data_filepath = "data/"
elev_file_list = ['elevation.tif']
elevation = GeeImages(data_filepath, elev_file_list)
moz_aug_df3 = elevation.add_all_images(moz_aug_df3)


# Add the land cover
data_filepath = "data/"
lc_file_list = ['land_cover_mode.tif']
land_cover = GeeImages(data_filepath, lc_file_list)
moz_aug_df3 = land_cover.add_all_images(moz_aug_df3)


# Add surface soil moisture
soil_filepath = "data/soil_moisture/"
ssm_file_list = [f'ssm_t-{i}.tif' for i in range(28, 0, -3)]
ssm = GeeImages(soil_filepath, ssm_file_list)
moz_aug_df3 = ssm.add_all_images(moz_aug_df3)


# Duplicate constant features into temporal features
moz_aug_df3 = copy_col(moz_aug_df3, column='elevation', rep=30)
moz_aug_df3 = copy_col(moz_aug_df3, column='land_cover_mode', rep=30)


# Duplicate temporal features
moz_aug_df3 = copy_time_col(moz_aug_df3, column_stem='ssm', start=28, end=0, gap=2, step=-3)



# Add the slope
data_filepath = "data/"
slope_file_list = ['slope.tif']
slope = GeeImages(data_filepath, slope_file_list)
moz_aug_df3 = slope.add_all_images(moz_aug_df3)


# Duplicate constant features into temporal features
moz_aug_df3 = copy_col(moz_aug_df3, column='slope', rep=30)


# Impute NaN with mean
moz_aug_df3 = moz_aug_df3.fillna(moz_aug_df3.mean())


# Order the features with repsect to timesetp
header = order_feats(moz_aug_df3)

write_file(moz_aug_df3, header=header, filename="data/moz_aug_df3.csv")





##############   Malwai 


# df = pd.read_csv("data/training_set.csv")


# train = pd.read_csv("data/Train.csv")


def plot_test_flood(train):
        
    # get all unique x, y coordinates
    _x = train['X'].round(2).unique()
    _y = train['Y'].round(2).unique()
    print(len(_x), len(_y))
    
    #and create their all possible combinations
    _xn = np.meshgrid(_x, _y)[0]
    _yn = np.meshgrid(_x, _y)[1]
    
    all_xy = np.dstack([_xn, _yn]).reshape(-1, 2)
    
    
    #with this combinations create an empty df and merge it with the original one by unique (x, y) tuples
    train_full = pd.DataFrame()
    train_full.loc[:, 'X'] = all_xy[:, 0]
    train_full.loc[:, 'Y'] = all_xy[:, 1]
    train_full.loc[:, 'id'] = train_full['X'].astype(str) + '_' + train_full['Y'].astype(str)
    id2ind = dict(zip(train_full.loc[:, 'id'].values, train_full.loc[:, 'id'].factorize()[0]))
    train_full.loc[:, 'id'] = train_full.loc[:, 'id'].map(id2ind)
    
    train.loc[:, 'id'] = train['X'].astype(str) + '_' + train['Y'].astype(str)
    train.loc[:, 'id'] = train.loc[:, 'id'].map(id2ind)
    del train['X'], train['Y']
    
    train_full = train_full.merge(train, on=['id'], how='left').sort_values(['Y', 'X'], ascending=[False, True]).reset_index(drop=True)
    del train_full['id']
    
    grid = all_xy
    rows = grid.shape[0]
    cols = grid.shape[1]
    cols = len(list(np.arange(xmin, xmax, 0.01)))
    rows = len(list(np.arange(ymin, ymax, 0.01)))
    
    img_target = train_full['target'].values.reshape(cols,rows)
    
    plt.imshow(img_target, cmap='Blues')
    plt.title('2015 target')
    return plt.show()



# df = moz_orig_df.copy()

# df['X'] = gpd.GeoSeries(df.centroid).x

# df['Y'] = gpd.GeoSeries(df.centroid).y

# df['target'] = moz_orig_df.target

# # ax = pd.DataFrame(df).plot.hexbin(x='coord_x',
# #                     y='coord_y',
# #                     C='target',
# #                     #reduce_C_function=np.sum,
# #                     #gridsize=10,
# #                     cmap="Blues")

# plot_test_flood(df)


# test_df = df.iloc[:len(train), :]



# test_df.iloc[:, 5]



# ============================================================================

test_zipfile = "zip:///home/leo/Desktop/ml_flood_prediction/data/mwi_orig_flood_vec.zip"

test_gdf = gpd.read_file(test_zipfile)



# # Total bounds for image
# xmin,ymin,xmax,ymax = zip(test_gdf.total_bounds)



# Total bounds for image
xmin,ymin,xmax,ymax = 34.50, -17.128481, 35.91677138449524, -13.50, 


# Create a grid
mwi_orig_grid = make_grid(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
# Plot the grid
plot_grid(flood=test_gdf, grid=mwi_orig_grid)


# # Use new grid
# mwi_new_grid = inside_border(mwi_orig_grid, country='Malawi')
# plot_grid(flood=test_gdf, grid=mwi_new_grid)


# Get flood dataframe with fraction of grids flooded
mwi_orig_df = flood_frac_df(gdf=test_gdf, grid=mwi_orig_grid)
# Plot the floods
plot_flood(mwi_orig_df, mwi_orig_grid, xmin, xmax, ymin, ymax)



# Add the total precipitation
total_precip_filepath = "data/mwi_precip/"
tr_file_list = [f'total_precip_t-{i}.tif' for i in range(30, 0, -1)]
total_precip = GeeImages(total_precip_filepath, tr_file_list)
mwi_orig_df = total_precip.add_all_images(mwi_orig_df)

# Add the elevation
data_filepath = "data/"
elev_file_list = ['mwi_elevation.tif']
elevation = GeeImages(data_filepath, elev_file_list)
mwi_orig_df = elevation.add_all_images(mwi_orig_df)


# Add the land cover
data_filepath = "data/"
lc_file_list = ['mwi_land_cover_mode.tif']
land_cover = GeeImages(data_filepath, lc_file_list)
mwi_orig_df = land_cover.add_all_images(mwi_orig_df)


# Add surface soil moisture
soil_filepath = "data/mwi_soil_moisture/"
ssm_file_list = [f'ssm_t-{i}.tif' for i in range(28, 0, -3)]
ssm = GeeImages(soil_filepath, ssm_file_list)
mwi_orig_df = ssm.add_all_images(mwi_orig_df)


# Add the land cover
data_filepath = "data/"
slope_file_list = ['mwi_slope.tif']
slope = GeeImages(data_filepath, slope_file_list)
mwi_orig_df = slope.add_all_images(mwi_orig_df)


# Duplicate constant features into temporal features
mwi_orig_df = copy_col(mwi_orig_df, column='mwi_elevation', rep=30)
mwi_orig_df = copy_col(mwi_orig_df, column='mwi_slope', rep=30)
mwi_orig_df = copy_col(mwi_orig_df, column='mwi_land_cover_mode', rep=30)


# Duplicate temporal features
mwi_orig_df = copy_time_col(mwi_orig_df, column_stem='ssm', start=28, end=0, gap=2, step=-3)

# Order the features with repsect to timesetp
header = order_feats(mwi_orig_df)


# Impute NaN with mean
mwi_orig_df = mwi_orig_df.fillna(mwi_orig_df.mean())


write_file(mwi_orig_df, header=header, filename="data/mwi_orig_df.csv")

