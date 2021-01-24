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

