#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 13:53:00 2020

@author: leo
"""

from gee_images import GeeImages

import numpy as np
import pandas as pd
from rasterstats import zonal_stats


# Read in Zindi dataset
df = pd.read_csv('data/zindi/Train (2).csv')

# Convert to Geo Dataframe
df['geometry'] = coords_to_geom(df)
test_gdf = gpd.GeoDataFrame(df, crs="EPSG:4326")


import geopandas as gpd

#T = gpd.read_file("data/soil_organic_carbon-mean.tif")


import rasterio
#crs = rasterio.crs.CRS({"init": "epsg:4326"})    # or whatever CRS you know the image is in    

src = rasterio.open('data/new.tif')


# # Change the CRS to EPSG 4326
# test_gdf.to_crs("+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs")


ge = GeeImages(data_filepath, lc_file_list)


T = ge.add_image(test_gdf, file='data/soil_organic_carbon-mean.tif')

file='data/new.tif'
image = rasterio.open(file)
band1 = image.read(1)
affine = image.transform
#gdf = gpd.GeoDataFrame(df)

# test_gdf.to_file('t.shp')

# shp = 't.shp'



stat = zonal_stats(test_gdf, file, stats=['mean'], band=1)



g = gpd.GeoDataFrame(band1)

# Add the land cover
data_filepath = "data/"
lc_file_list = ['soil_organic_carbon-mean.tif']
land_cover = GeeImages(data_filepath, lc_file_list)
test_gdf = land_cover.add_all_images(test_gdf)









with rasterio.open('data/soil_organic_carbon-mean.tif', mode='r+') as src:
    #src.transform = transform
    src.crs = crs
    #array = src.read(1)
    
with rasterio.open('example.tif', 'w', **profile) as dst:
        dst.write(array.astype(rasterio.uint8), 1)

import matplotlib.pyplot as plt
plt.imshow(array, cmap='pink')
plt.show() 




import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

dst_crs = 'EPSG:4326'

with rasterio.open('data/soil_organic_carbon-mean.tif') as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height
    })

    with rasterio.open('/tmp/RGB.byte.wgs84.tif', 'w', **kwargs) as dst:
        for i in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, i),
                destination=rasterio.band(dst, i),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)