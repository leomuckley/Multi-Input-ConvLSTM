#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points

from gee_images import GeeImages



def make_grid(xmin, ymin, xmax, ymax):
    """ 
    Function for creating a grid using polygon objects. 
    
    Parameters
    ----------
    xmin : float, geo-coordinate
    ymin : float, geo-coordinate
    xmax : float, geo-coordinate
    ymax : float, geo-coordinate
    
    Returns
    -------
    grid : GeoDataFrame
    
    """
    length = 0.01
    wide = 0.01

    cols = list(np.arange(xmin, xmax, wide))
    rows = list(np.arange(ymin, ymax, length))
    rows = rows[::-1]
    
    polygons = []
    for x in cols:
        for y in rows:
            polygons.append( Polygon([(x,y), (x+wide, y), (x+wide, y-length), (x, y-length)]) )
    
    grid = gpd.GeoDataFrame({'geometry':polygons})
    return grid 


def plot_grid(flood, grid):
    """ 
    Plot grid over flood image. 
    
    Parameters
    ----------
    flood : GeoDataFrame
    grid  : GeoDataFrame
        
    """
    ax = grid.plot(color='yellow');
    return flood.plot(ax=ax, color='black', alpha=0.5);


def flood_frac_df(gdf, grid):
    """ 
    Function for calculating the fraction of each square in the grid
    that contains flooding. Each square conatined in the grid will then
    return a value in the interval between 0 and 1.
    
    Parameters
    ----------
    gdf  : GeoDataFrame
    grid : GeoDataFrame
    
    Returns
    -------
    df   : DataFrame
           The resuling DataFrame will contain the fraction of flooding,
           for each of the original coordinates.
    
    """
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


def dist_to_water(df, zipfile, geom1_col='geometry'):
    # import data from http://www.diva-gis.org/datadown
    #zipfile = "zip:///home/leo/Desktop/ml_flood_prediction/data/inland_water/MOZ_wat.zip"
    mwi_water = gpd.read_file(zipfile)
    water_union = mwi_water.unary_union
    dist_list =[]
    for index, row in df.iterrows():
        nearest = nearest_points(row[geom1_col], water_union)[1]
        dist = row[geom1_col].distance(nearest)
        dist_list.append(dist)
    
    # Return column
    return dist_list


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


def copy_time_fw_col(df_, column_stem, start, end, step, gap):
    df = df_.copy()
    for i in range(start, end, step):
        for j in range(i, i-gap, -1):
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


def return_coords(df_):
    df = df_.copy()
    def get_coords(row):
        row['X'] = row['centroid'].x
        row['Y'] = row['centroid'].y     
        return row
    return df.apply(get_coords, axis=1)



def coords_to_geom(df_):
    df = df_.copy()
    geom = [Point(df.X[i], df.Y[i]) for i in range(len(df))]
    #df['geometry'] = geom
    return geom

################      Original Mozambique GEE Image      ######################



train_zipfile = "zip:///home/leo/Desktop/ml_flood_prediction/data/floods_13-03-2020/moz_flood.zip"

gdf = gpd.read_file(train_zipfile)

moz_water = "zip:///home/leo/Desktop/ml_flood_prediction/data/inland_water/MOZ_wat.zip"



######

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

# Add distance to inland water
distance = dist_to_water(moz_orig_df, moz_water)
moz_orig_df['dist_to_water'] = distance

# Add the total precipitation
total_precip_filepath = "data/new_moz_floods_2019-03-13/"
tr_file_list = [f'total_precip_t-{i}.tif' for i in range(30, 0, -1)]
total_precip = GeeImages(total_precip_filepath, tr_file_list)
moz_orig_df = total_precip.add_all_images(moz_orig_df)


# Add the max precipitation
max_precip_filepath = "data/new_moz_floods_2019-03-13/"
mr_file_list = [f'max_precip_t-{i}.tif' for i in range(30, 0, -1)]
max_precip = GeeImages(max_precip_filepath, mr_file_list)
moz_orig_df = max_precip.add_all_images(moz_orig_df)

# Add the std precipitation
std_precip_filepath = "data/new_moz_floods_2019-03-13/"
stdr_file_list = [f'std_precip_t-{i}.tif' for i in range(30, 0, -1)]
std_precip = GeeImages(std_precip_filepath, stdr_file_list)
moz_orig_df = std_precip.add_all_images(moz_orig_df)


# Add the elevation
data_filepath = "data/floods_13-03-2020/moz_feats/"
elev_file_list = ['elevation.tif']
elevation = GeeImages(data_filepath, elev_file_list)
moz_orig_df = elevation.add_all_images(moz_orig_df)


# Add the land cover
data_filepath = "data/floods_13-03-2020/moz_feats/"
lc_file_list = ['land_cover_mode.tif']
land_cover = GeeImages(data_filepath, lc_file_list)
moz_orig_df = land_cover.add_all_images(moz_orig_df)


# Add the slope
data_filepath = "data/floods_13-03-2020/moz_feats/"
slope_file_list = ['slope.tif']
slope = GeeImages(data_filepath, slope_file_list)
moz_orig_df = slope.add_all_images(moz_orig_df)


# Add surface soil moisture
soil_filepath = "data/floods_13-03-2020/moz_feats/"
ssm_file_list = [f'ssm_t-{i}.tif' for i in range(30, 0, -3)]
ssm = GeeImages(soil_filepath, ssm_file_list)
moz_orig_df = ssm.add_all_images(moz_orig_df)

# Add the soil organic carbon mean
data_filepath = "data/floods_13-03-2020/moz_feats/"
clay_file_list = ['clay_mean.tif']
clay = GeeImages(data_filepath, clay_file_list)
moz_orig_df = clay.add_all_images(moz_orig_df)


# Add the slope
data_filepath = "data/floods_13-03-2020/moz_feats/"
slope_file_list = ['slope.tif']
slope = GeeImages(data_filepath, slope_file_list)
moz_orig_df = slope.add_all_images(moz_orig_df)


# Add the soil organic carbon mean
data_filepath = "data/floods_13-03-2020/moz_feats/"
soc_file_list = ['soc_mean.tif']
soc = GeeImages(data_filepath, soc_file_list)
moz_orig_df = soc.add_all_images(moz_orig_df)


# add coordinates from centroids
moz_orig_df = return_coords(moz_orig_df)


# Duplicate constant features into temporal features
moz_orig_df = copy_col(moz_orig_df, column='elevation', rep=30)
moz_orig_df = copy_col(moz_orig_df, column='slope', rep=30)
moz_orig_df = copy_col(moz_orig_df, column='land_cover_mode', rep=30)
moz_orig_df = copy_col(moz_orig_df, column='X', rep=30)
moz_orig_df = copy_col(moz_orig_df, column='Y', rep=30)
moz_orig_df = copy_col(moz_orig_df, column='dist_to_water', rep=30)
moz_orig_df = copy_col(moz_orig_df, column='soc_mean', rep=30)




# Duplicate temporal features
moz_orig_df = copy_time_fw_col(moz_orig_df, column_stem='ssm', start=30, end=0, gap=3, step=-3)

# Order the features with repsect to timesetp
header = order_feats(moz_orig_df)

"""
Maybe delete the NaN inputs.
"""

# Impute NaN with mean
#moz_orig_df = moz_orig_df.fillna(moz_orig_df.mean())
moz_orig_df = moz_orig_df.dropna()


#write_file(moz_orig_df, header=header, filename="data/moz_13mar_df.csv")
write_file(moz_orig_df, header=header, filename="data/moz_13mar_dropna_df.csv")



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

# Add distance to inland water
distance = dist_to_water(moz_aug_df1, moz_water)
moz_aug_df1['dist_to_water'] = distance

# Add the total precipitation
total_precip_filepath = "data/new_moz_floods_2019-03-13/"
tr_file_list = [f'total_precip_t-{i}.tif' for i in range(30, 0, -1)]
total_precip = GeeImages(total_precip_filepath, tr_file_list)
moz_aug_df1 = total_precip.add_all_images(moz_aug_df1)


# Add the max precipitation
max_precip_filepath = "data/new_moz_floods_2019-03-13/"
mr_file_list = [f'max_precip_t-{i}.tif' for i in range(30, 0, -1)]
max_precip = GeeImages(max_precip_filepath, mr_file_list)
moz_aug_df1 = max_precip.add_all_images(moz_aug_df1)

# Add the std precipitation
std_precip_filepath = "data/new_moz_floods_2019-03-13/"
stdr_file_list = [f'std_precip_t-{i}.tif' for i in range(30, 0, -1)]
std_precip = GeeImages(std_precip_filepath, stdr_file_list)
moz_aug_df1 = std_precip.add_all_images(moz_aug_df1)

# Add the elevation
data_filepath = "data/floods_13-03-2020/moz_feats/"
elev_file_list = ['elevation.tif']
elevation = GeeImages(data_filepath, elev_file_list)
moz_aug_df1 = elevation.add_all_images(moz_aug_df1)


# Add the land cover
data_filepath = "data/floods_13-03-2020/moz_feats/"
lc_file_list = ['land_cover_mode.tif']
land_cover = GeeImages(data_filepath, lc_file_list)
moz_aug_df1 = land_cover.add_all_images(moz_aug_df1)


# Add surface soil moisture
soil_filepath = "data/floods_13-03-2020/moz_feats/"
ssm_file_list = [f'ssm_t-{i}.tif' for i in range(30, 0, -3)]
ssm = GeeImages(soil_filepath, ssm_file_list)
moz_aug_df1 = ssm.add_all_images(moz_aug_df1)

# Add the slope
data_filepath = "data/floods_13-03-2020/moz_feats/"
slope_file_list = ['slope.tif']
slope = GeeImages(data_filepath, slope_file_list)
moz_aug_df1 = slope.add_all_images(moz_aug_df1)


# Add the soil organic carbon mean
data_filepath = "data/floods_13-03-2020/moz_feats/"
soc_file_list = ['soc_mean.tif']
soc = GeeImages(data_filepath, soc_file_list)
moz_aug_df1 = soc.add_all_images(moz_aug_df1)



# add coordinates from centroids
moz_aug_df1 = return_coords(moz_aug_df1)

# Duplicate constant features into temporal features
moz_aug_df1 = copy_col(moz_aug_df1, column='elevation', rep=30)
moz_aug_df1 = copy_col(moz_aug_df1, column='slope', rep=30)
moz_aug_df1 = copy_col(moz_aug_df1, column='land_cover_mode', rep=30)
moz_aug_df1 = copy_col(moz_aug_df1, column='X', rep=30)
moz_aug_df1 = copy_col(moz_aug_df1, column='Y', rep=30)
moz_aug_df1 = copy_col(moz_aug_df1, column='dist_to_water', rep=30)
moz_aug_df1 = copy_col(moz_aug_df1, column='soc_mean', rep=30)


# Duplicate temporal features
moz_aug_df1 = copy_time_fw_col(moz_aug_df1, column_stem='ssm', start=30, end=0, gap=3, step=-3)



# Impute NaN with mean
#moz_aug_df1 = moz_aug_df1.fillna(moz_aug_df1.mean())
moz_aug_df1 = moz_aug_df1.dropna()

# Order the features with repsect to timesetp
header = order_feats(moz_aug_df1)

#write_file(moz_aug_df1, header=header, filename="data/moz_13mar_df1.csv")
write_file(moz_aug_df1, header=header, filename="data/moz_13mar_dropna_df1.csv")


###### Data augmentation 2: stagger y-axis by 0.5

# Total bounds for image
xmin,ymin,xmax,ymax = 33.50, -20.545, 35.50, -19.055

# Create a grid
moz_orig_grid = make_grid(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
# Plot the grid
plot_grid(flood=gdf, grid=moz_orig_grid)

# # Use new grid
# moz_new_grid = inside_border(moz_orig_grid, country='Mozambique')
# plot_grid(flood=gdf, grid=moz_new_grid)


# Get flood dataframe with fraction of grids flooded
moz_aug_df2 = flood_frac_df(gdf=gdf, grid=moz_orig_grid)
# Plot the floods
plot_flood(moz_aug_df2, moz_orig_grid, xmin, xmax, ymin, ymax)

# Add distance to inland water
distance = dist_to_water(moz_aug_df2, moz_water)
moz_aug_df2['dist_to_water'] = distance


# Add the total precipitation
total_precip_filepath = "data/new_moz_floods_2019-03-13/"
tr_file_list = [f'total_precip_t-{i}.tif' for i in range(30, 0, -1)]
total_precip = GeeImages(total_precip_filepath, tr_file_list)
moz_aug_df2 = total_precip.add_all_images(moz_aug_df2)


# Add the max precipitation
max_precip_filepath = "data/new_moz_floods_2019-03-13/"
mr_file_list = [f'max_precip_t-{i}.tif' for i in range(30, 0, -1)]
max_precip = GeeImages(max_precip_filepath, mr_file_list)
moz_aug_df2 = max_precip.add_all_images(moz_aug_df2)


# Add the std precipitation
std_precip_filepath = "data/new_moz_floods_2019-03-13/"
stdr_file_list = [f'std_precip_t-{i}.tif' for i in range(30, 0, -1)]
std_precip = GeeImages(std_precip_filepath, stdr_file_list)
moz_aug_df2 = std_precip.add_all_images(moz_aug_df2)

# Add the elevation
data_filepath = "data/floods_13-03-2020/moz_feats/"
elev_file_list = ['elevation.tif']
elevation = GeeImages(data_filepath, elev_file_list)
moz_aug_df2 = elevation.add_all_images(moz_aug_df2)


# Add the land cover
data_filepath = "data/floods_13-03-2020/moz_feats/"
lc_file_list = ['land_cover_mode.tif']
land_cover = GeeImages(data_filepath, lc_file_list)
moz_aug_df2 = land_cover.add_all_images(moz_aug_df2)


# Add surface soil moisture
soil_filepath = "data/floods_13-03-2020/moz_feats/"
ssm_file_list = [f'ssm_t-{i}.tif' for i in range(30, 0, -3)]
ssm = GeeImages(soil_filepath, ssm_file_list)
moz_aug_df2 = ssm.add_all_images(moz_aug_df2)


# Add the slope
data_filepath = "data/floods_13-03-2020/moz_feats/"
slope_file_list = ['slope.tif']
slope = GeeImages(data_filepath, slope_file_list)
moz_aug_df2 = slope.add_all_images(moz_aug_df2)


# Add the soil organic carbon mean
data_filepath = "data/floods_13-03-2020/moz_feats/"
soc_file_list = ['soc_mean.tif']
soc = GeeImages(data_filepath, soc_file_list)
moz_aug_df2 = soc.add_all_images(moz_aug_df2)


# add coordinates from centroids
moz_aug_df2 = return_coords(moz_aug_df2)

# Duplicate constant features into temporal features
moz_aug_df2 = copy_col(moz_aug_df2, column='elevation', rep=30)
moz_aug_df2 = copy_col(moz_aug_df2, column='slope', rep=30)
moz_aug_df2 = copy_col(moz_aug_df2, column='land_cover_mode', rep=30)
moz_aug_df2 = copy_col(moz_aug_df2, column='X', rep=30)
moz_aug_df2 = copy_col(moz_aug_df2, column='Y', rep=30)
moz_aug_df2 = copy_col(moz_aug_df2, column='dist_to_water', rep=30)
moz_aug_df2 = copy_col(moz_aug_df2, column='soc_mean', rep=30)


# Duplicate temporal features
moz_aug_df2 = copy_time_fw_col(moz_aug_df2, column_stem='ssm', start=30, end=0, gap=3, step=-3)


# Impute NaN with mean
#moz_aug_df2 = moz_aug_df2.fillna(moz_aug_df2.mean())

moz_aug_df2 = moz_aug_df2.dropna()

# Order the features with repsect to timesetp
header = order_feats(moz_aug_df2)

#write_file(moz_aug_df2, header=header, filename="data/moz_13mar_df2.csv")
write_file(moz_aug_df2, header=header, filename="data/moz_13mar_dropna_df2.csv")

###### Data augmentation 3: stagger the x-axis by 0.5 and the y-axis by 0.5

# Total bounds for image
xmin,ymin,xmax,ymax = 33.505, -20.545, 35.495, -19.055

# Create a grid
moz_orig_grid = make_grid(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
# Plot the grid
plot_grid(flood=gdf, grid=moz_orig_grid)


# # Use new grid
# moz_new_grid = inside_border(moz_orig_grid, country='Mozambique')
# plot_grid(flood=gdf, grid=moz_new_grid)


# Get flood dataframe with fraction of grids flooded
moz_aug_df3 = flood_frac_df(gdf=gdf, grid=moz_orig_grid)
# Plot the floods
plot_flood(moz_aug_df3, moz_orig_grid, xmin, xmax, ymin, ymax)


# Add distance to inland water
distance = dist_to_water(moz_aug_df3, moz_water)
moz_aug_df3['dist_to_water'] = distance


# Add the total precipitation
total_precip_filepath = "data/new_moz_floods_2019-03-13/"
tr_file_list = [f'total_precip_t-{i}.tif' for i in range(30, 0, -1)]
total_precip = GeeImages(total_precip_filepath, tr_file_list)
moz_aug_df3 = total_precip.add_all_images(moz_aug_df3)


# Add the max precipitation
max_precip_filepath = "data/new_moz_floods_2019-03-13/"
mr_file_list = [f'max_precip_t-{i}.tif' for i in range(30, 0, -1)]
max_precip = GeeImages(max_precip_filepath, mr_file_list)
moz_aug_df3 = max_precip.add_all_images(moz_aug_df3)


# Add the std precipitation
std_precip_filepath = "data/new_moz_floods_2019-03-13/"
stdr_file_list = [f'std_precip_t-{i}.tif' for i in range(30, 0, -1)]
std_precip = GeeImages(std_precip_filepath, stdr_file_list)
moz_aug_df3 = std_precip.add_all_images(moz_aug_df3)

# Add the elevation
data_filepath = "data/floods_13-03-2020/moz_feats/"
elev_file_list = ['elevation.tif']
elevation = GeeImages(data_filepath, elev_file_list)
moz_aug_df3 = elevation.add_all_images(moz_aug_df3)


# Add the land cover
data_filepath = "data/floods_13-03-2020/moz_feats/"
lc_file_list = ['land_cover_mode.tif']
land_cover = GeeImages(data_filepath, lc_file_list)
moz_aug_df3 = land_cover.add_all_images(moz_aug_df3)


# Add surface soil moisture
soil_filepath = "data/floods_13-03-2020/moz_feats/"
ssm_file_list = [f'ssm_t-{i}.tif' for i in range(30, 0, -3)]
ssm = GeeImages(soil_filepath, ssm_file_list)
moz_aug_df3 = ssm.add_all_images(moz_aug_df3)



# Add the slope
data_filepath = "data/floods_13-03-2020/moz_feats/"
slope_file_list = ['slope.tif']
slope = GeeImages(data_filepath, slope_file_list)
moz_aug_df3 = slope.add_all_images(moz_aug_df3)


# Add the soil organic carbon mean
data_filepath = "data/floods_13-03-2020/moz_feats/"
soc_file_list = ['soc_mean.tif']
soc = GeeImages(data_filepath, soc_file_list)
moz_aug_df3 = soc.add_all_images(moz_aug_df3)



# add coordinates from centroids
moz_aug_df3 = return_coords(moz_aug_df3)

# Duplicate constant features into temporal features
moz_aug_df3 = copy_col(moz_aug_df3, column='elevation', rep=30)
moz_aug_df3 = copy_col(moz_aug_df3, column='slope', rep=30)
moz_aug_df3 = copy_col(moz_aug_df3, column='land_cover_mode', rep=30)
moz_aug_df3 = copy_col(moz_aug_df3, column='X', rep=30)
moz_aug_df3 = copy_col(moz_aug_df3, column='Y', rep=30)
moz_aug_df3 = copy_col(moz_aug_df3, column='dist_to_water', rep=30)
moz_aug_df3 = copy_col(moz_aug_df3, column='soc_mean', rep=30)


# Duplicate temporal features
moz_aug_df3 = copy_time_fw_col(moz_aug_df3, column_stem='ssm', start=30, end=0, gap=3, step=-3)


# Impute NaN with mean
#moz_aug_df3 = moz_aug_df3.fillna(moz_aug_df3.mean())
moz_aug_df3 = moz_aug_df3.dropna()


# Order the features with repsect to timesetp
header = order_feats(moz_aug_df3)

#write_file(moz_aug_df3, header=header, filename="data/moz_13mar_df3.csv")
write_file(moz_aug_df3, header=header, filename="data/moz_13mar_dropna_df3.csv")


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





# ============================================================================

test_zipfile = "zip:///home/leo/Desktop/ml_flood_prediction/data/floods_13-03-2020/mwi_flood.zip"

test_gdf = gpd.read_file(test_zipfile)

mwi_water = "zip:///home/leo/Desktop/ml_flood_prediction/data/inland_water/MWI_wat.zip"


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



# Add distance to inland water
distance = dist_to_water(mwi_orig_df, mwi_water)
mwi_orig_df['dist_to_water'] = distance


# Add the total precipitation
total_precip_filepath = "data/new_mwi_floods_2019-03-13/"
tr_file_list = [f'total_precip_t-{i}.tif' for i in range(30, 0, -1)]
total_precip = GeeImages(total_precip_filepath, tr_file_list)
mwi_orig_df = total_precip.add_all_images(mwi_orig_df)


# Add the max precipitation
max_precip_filepath = "data/new_mwi_floods_2019-03-13/"
mr_file_list = [f'max_precip_t-{i}.tif' for i in range(30, 0, -1)]
max_precip = GeeImages(max_precip_filepath, mr_file_list)
mwi_orig_df = max_precip.add_all_images(mwi_orig_df)


# Add the std precipitation
std_precip_filepath = "data/new_mwi_floods_2019-03-13/"
stdr_file_list = [f'std_precip_t-{i}.tif' for i in range(30, 0, -1)]
std_precip = GeeImages(std_precip_filepath, stdr_file_list)
mwi_orig_df = std_precip.add_all_images(mwi_orig_df)

# Add the elevation
data_filepath = "data/floods_13-03-2020/mwi_feats/"
elev_file_list = ['elevation.tif']
elevation = GeeImages(data_filepath, elev_file_list)
mwi_orig_df = elevation.add_all_images(mwi_orig_df)


# Add the land cover
data_filepath = "data/floods_13-03-2020/mwi_feats/"
lc_file_list = ['land_cover_mode.tif']
land_cover = GeeImages(data_filepath, lc_file_list)
mwi_orig_df = land_cover.add_all_images(mwi_orig_df)


# Add surface soil moisture
soil_filepath = "data/floods_13-03-2020/mwi_feats/"
ssm_file_list = [f'ssm_t-{i}.tif' for i in range(30, 0, -3)]
ssm = GeeImages(soil_filepath, ssm_file_list)
mwi_orig_df = ssm.add_all_images(mwi_orig_df)


# Add the slope
data_filepath = "data/floods_13-03-2020/mwi_feats/"
slope_file_list = ['slope.tif']
slope = GeeImages(data_filepath, slope_file_list)
mwi_orig_df = slope.add_all_images(mwi_orig_df)

# Add the soil organic carbon mean
data_filepath = "data/floods_13-03-2020/mwi_feats/"
soc_file_list = ['clay_mean.tif']
soc = GeeImages(data_filepath, soc_file_list)
mwi_orig_df = soc.add_all_images(mwi_orig_df)


# Add the soil organic carbon mean
data_filepath = "data/floods_13-03-2020/mwi_feats/"
soc_file_list = ['soc_mean.tif']
soc = GeeImages(data_filepath, soc_file_list)
mwi_orig_df = soc.add_all_images(mwi_orig_df)


# add coordinates from centroids
mwi_orig_df = return_coords(mwi_orig_df)


# Duplicate constant features into temporal features
mwi_orig_df = copy_col(mwi_orig_df, column='elevation', rep=30)
mwi_orig_df = copy_col(mwi_orig_df, column='clay_mean', rep=30)
mwi_orig_df = copy_col(mwi_orig_df, column='slope', rep=30)
mwi_orig_df = copy_col(mwi_orig_df, column='land_cover_mode', rep=30)
mwi_orig_df = copy_col(mwi_orig_df, column='X', rep=30)
mwi_orig_df = copy_col(mwi_orig_df, column='Y', rep=30)
mwi_orig_df = copy_col(mwi_orig_df, column='dist_to_water', rep=30)
mwi_orig_df = copy_col(mwi_orig_df, column='soc_mean', rep=30)


# Duplicate temporal features
mwi_orig_df = copy_time_fw_col(mwi_orig_df, column_stem='ssm', start=30, end=0, gap=3, step=-3)

# Order the features with repsect to timesetp
header = order_feats(mwi_orig_df)


# Impute NaN with mean
mwi_fill_df = mwi_orig_df.fillna(mwi_orig_df.mean())
mwi_drop_df = mwi_orig_df.dropna()


write_file(mwi_drop_df, header=header, filename="data/mwi_dropna_13mar_df.csv")

write_file(mwi_fill_df, header=header, filename="data/mwi_13mar_df.csv")




###############################################################################

# Read in Zindi dataset
df = pd.read_csv('data/zindi/Train.csv')


# Convert to Geo Dataframe
df['geometry'] = coords_to_geom(df)
test_gdf = gpd.GeoDataFrame(df)

mwi_water = "zip:///home/leo/Desktop/ml_flood_prediction/data/inland_water/MWI_wat.zip"


# Add distance to inland water
distance = dist_to_water(test_gdf, mwi_water)
test_gdf['dist_to_water'] = distance


# Add the total precipitation
total_precip_filepath = "data/new_mwi_floods_2019-03-13/"
tr_file_list = [f'total_precip_t-{i}.tif' for i in range(30, 0, -1)]
total_precip = GeeImages(total_precip_filepath, tr_file_list)
test_gdf = total_precip.add_all_images(test_gdf)


# Add the max precipitation
max_precip_filepath = "data/new_mwi_floods_2019-03-13/"
mr_file_list = [f'max_precip_t-{i}.tif' for i in range(30, 0, -1)]
max_precip = GeeImages(max_precip_filepath, mr_file_list)
test_gdf = max_precip.add_all_images(test_gdf)


# Add the std precipitation
std_precip_filepath = "data/new_mwi_floods_2019-03-13/"
stdr_file_list = [f'std_precip_t-{i}.tif' for i in range(30, 0, -1)]
std_precip = GeeImages(std_precip_filepath, stdr_file_list)
test_gdf = std_precip.add_all_images(test_gdf)

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


# Add the slope
data_filepath = "data/floods_13-03-2020/mwi_feats/"
slope_file_list = ['slope.tif']
slope = GeeImages(data_filepath, slope_file_list)
test_gdf = slope.add_all_images(test_gdf)


# Add the soil organic carbon mean
data_filepath = "data/floods_13-03-2020/mwi_feats/"
soc_file_list = ['soc_mean.tif']
soc = GeeImages(data_filepath, soc_file_list)
test_gdf = soc.add_all_images(test_gdf)



# Duplicate constant features into temporal features
test_gdf = copy_col(test_gdf, column='elevation', rep=30)
test_gdf = copy_col(test_gdf, column='slope', rep=30)
test_gdf = copy_col(test_gdf, column='land_cover_mode', rep=30)
test_gdf = copy_col(test_gdf, column='X', rep=30)
test_gdf = copy_col(test_gdf, column='Y', rep=30)
test_gdf = copy_col(test_gdf, column='dist_to_water', rep=30)
test_gdf = copy_col(test_gdf, column='soc_mean', rep=30)


# Duplicate temporal features
test_gdf = copy_time_col(test_gdf, column_stem='ssm', start=28, end=0, gap=2, step=-3)

test_gdf['target'] = test_gdf['target_2015']
final_cols = set(df.columns).intersection(set(test_gdf.columns))

# Order the features with repsect to timesetp
header = order_feats(test_gdf)

header = [i for i in header if "precip " not in i]

header = [i for i in header if ("-0" not in i and "--" not in i)]

# Impute NaN with mean
#test_gdf = test_gdf.fillna(test_gdf.mean())

#final_gdf = test_gdf[header]

write_file(test_gdf, header=header, filename="data/zindi_orig_df.csv")



########  Kenya Flood for testing #####



new_zipfile = "zip:///home/leo/Desktop/ml_flood_prediction/data/floods_06-05-2020/ky_flood_vec.zip"

new_gdf = gpd.read_file(new_zipfile)

ken_water = "zip:///home/leo/Desktop/ml_flood_prediction/data/floods_06-05-2020/KEN_wat.zip"

#

# Total bounds for image
xmin,ymin,xmax,ymax = 38.12, 0.77, 39.50, 02.66

# Create a grid
ken_orig_grid = make_grid(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
# Plot the grid
plot_grid(flood=new_gdf, grid=ken_orig_grid)

# moz_new_grid = inside_border(moz_orig_grid, country='Mozambique')
# plot_grid(flood=gdf, grid=moz_new_grid)


# Get flood dataframe with fraction of grids flooded
ken_orig_df = flood_frac_df(gdf=new_gdf, grid=ken_orig_grid)
# Plot the floods
plot_flood(ken_orig_df, ken_orig_grid, xmin, xmax, ymin, ymax)



# Add distance to inland water
distance = dist_to_water(ken_orig_df, ken_water)
ken_orig_df['dist_to_water'] = distance


# Add the total precipitation
total_precip_filepath = "data/floods_06-05-2020/total_precip/"
tr_file_list = [f'total_precip_t-{i}.tif' for i in range(30, 0, -1)]
total_precip = GeeImages(total_precip_filepath, tr_file_list)
ken_orig_df = total_precip.add_all_images(ken_orig_df)


# Add the max precipitation
max_precip_filepath = "data/floods_06-05-2020/max_precip/"
mr_file_list = [f'max_precip_t-{i}.tif' for i in range(30, 0, -1)]
max_precip = GeeImages(max_precip_filepath, mr_file_list)
ken_orig_df = max_precip.add_all_images(ken_orig_df)


# Add the std precipitation
std_precip_filepath = "data/floods_06-05-2020/std_precip/"
stdr_file_list = [f'std_precip_t-{i}.tif' for i in range(30, 0, -1)]
std_precip = GeeImages(std_precip_filepath, stdr_file_list)
ken_orig_df = std_precip.add_all_images(ken_orig_df)

# Add the elevation
data_filepath = "data/floods_06-05-2020/"
elev_file_list = ['elevation.tif']
elevation = GeeImages(data_filepath, elev_file_list)
ken_orig_df = elevation.add_all_images(ken_orig_df)


# Add the land cover
data_filepath = "data/floods_06-05-2020/"
lc_file_list = ['land_cover_mode.tif']
land_cover = GeeImages(data_filepath, lc_file_list)
ken_orig_df = land_cover.add_all_images(ken_orig_df)


# Add surface soil moisture
soil_filepath = "data/floods_06-05-2020/soil_moisture/"
ssm_file_list = [f'ssm_t-{i}.tif' for i in range(30, 0, -3)]
ssm = GeeImages(soil_filepath, ssm_file_list)
ken_orig_df = ssm.add_all_images(ken_orig_df)


# Add the slope
data_filepath = "data/floods_06-05-2020/"
slope_file_list = ['slope.tif']
slope = GeeImages(data_filepath, slope_file_list)
ken_orig_df = slope.add_all_images(ken_orig_df)


# Add the soil organic carbon mean
data_filepath = "data/floods_06-05-2020/"
soc_file_list = ['soc_mean.tif']
soc = GeeImages(data_filepath, soc_file_list)
ken_orig_df = soc.add_all_images(ken_orig_df)


def copy_col(df_, column, rep):
    df = df_.copy()
    for i in range(rep, 0, -1):
        df[f'{column}_t-{i}'] = df[column]
    df = df.drop(column, axis=1)
    return df


def copy_ndvi(df, ndvi_list):
    df = df_.copy()
    s1 = df[f'ndvi_t-25']
    s2 = pd.Series([np.nan]len(s1))
    s3 = df[f'ndvi_t-9']
    D = pd.DataFrame([s1, s2, s3])
    interpolate = D.interpolate(axis=1)[1]
    for i in range(30, 25, -1):
        df[f'{ndvi}_t-{i}'] = df[f'ndvi_t-25']
    for i in range(25, 9, -1):
        df[f'{ndvi}_t-{i}'] = interpolate
    for i in range(9, 0, -1):
        df[f'{ndvi}_t-{i}'] = df[f'ndvi_t-9']
        
    return df
        
        

# add coordinates from centroids
ken_orig_df = return_coords(ken_orig_df)



# Duplicate constant features into temporal features
ken_orig_df = copy_col(ken_orig_df, column='elevation', rep=30)
ken_orig_df = copy_col(ken_orig_df, column='slope', rep=30)
ken_orig_df = copy_col(ken_orig_df, column='land_cover_mode', rep=30)
ken_orig_df = copy_col(ken_orig_df, column='X', rep=30)
ken_orig_df = copy_col(ken_orig_df, column='Y', rep=30)
ken_orig_df = copy_col(ken_orig_df, column='dist_to_water', rep=30)
ken_orig_df = copy_col(ken_orig_df, column='soc_mean', rep=30)


# Duplicate temporal features
ken_orig_df = copy_time_fw_col(ken_orig_df, column_stem='ssm', start=30, end=0, gap=3, step=-3)

# Order the features with repsect to timesetp
header = order_feats(ken_orig_df)


# Impute NaN with mean
#mwi_orig_df = mwi_orig_df.fillna(mwi_orig_df.mean())


# Impute NaN with mean
#moz_orig_df = moz_orig_df.fillna(moz_orig_df.mean())
ken_orig_df = ken_orig_df.dropna()


write_file(ken_orig_df, header=header, filename="data/ken_06may_dropna_df.csv")

