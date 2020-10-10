#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 17:00:29 2020

@author: leo
"""


from gee_images import GeeImages

import numpy as np
import pandas as pd
from rasterstats import zonal_stats
from shapely.geometry import Point

import geopandas as gpd



def coords_to_geom(df_):
    df = df_.copy()
    geom = [Point(df.X[i], df.Y[i]) for i in range(len(df))]
    #df['geometry'] = geom
    return geom

# Read in Zindi dataset
df = pd.read_csv('data/zindi/Train (2).csv')

# Convert to Geo Dataframe
df['geometry'] = coords_to_geom(df)
test_gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")


import os
import numpy as np
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
lidar_dem_path = os.path.join("data", "out.tif")


lidar_dem = rio.open(lidar_dem_path)
print(lidar_dem.meta)

dst_crs = 'EPSG:4326' # CRS for web meractor 

projected_lidar_dem_path = os.path.join("data", "new.tif")

with rio.open(lidar_dem_path) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rio.open(projected_lidar_dem_path, 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rio.band(src, i),
                destination=rio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)
            


# Define relative path to file
lidar_dem_path = os.path.join("data", "new.tif")

lidar_dem = rio.open(lidar_dem_path)
print(lidar_dem.meta)







