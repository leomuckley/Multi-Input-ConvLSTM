#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 17:06:48 2020

@author: leo
"""


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

moz_zipfile = "zip:///home/leo/Desktop/ml_flood_prediction/data/floods_13-03-2020/moz_flood.zip"
moz_gdf = gpd.read_file(moz_zipfile)
moz_geo = gpd.GeoDataFrame(moz_gdf)
xmin,ymin,xmax,ymax = 33.50, -20.60, 35.50, -19.00
moz_geo = moz_gdf.cx[xmin:xmax, ymin:ymax]


mwi_zipfile = "zip:///home/leo/Desktop/ml_flood_prediction/data/floods_13-03-2020/mwi_flood.zip"
mwi_gdf = gpd.read_file(mwi_zipfile)
mwi_geo = gpd.GeoDataFrame(mwi_gdf)
xmin, ymin, xmax, ymax = 34.26, -16.64, 35.86, -15.21
mwi_geo = mwi_gdf.cx[xmin:xmax, ymin:ymax]


ken_zipfile = "zip:///home/leo/Desktop/ml_flood_prediction/data/floods_06-05-2020/ky_flood_vec.zip"
ken_gdf = gpd.read_file(ken_zipfile)
ken_geo = gpd.GeoDataFrame(ken_gdf)
xmin,ymin,xmax,ymax = 38.12, 0.77, 39.50, 2.66
ken_geo = ken_gdf.cx[xmin:xmax, ymin:ymax]


import ee
import datetime

ee.Initialize()





def rainfall_images(new_geo, flood_start, folder):
    """ Export GEE images for rainfall.
    Example: flood_start = '2019-03-13' for mozambique.
    """
    
    trmm = ee.ImageCollection("TRMM/3B42")# new_shapefile = "mwi_3-18.shp"
    
    trmm = trmm.select(['precipitation'])

    # setting the Area of Interest (AOI)
    mwi_aoi = ee.Geometry.Rectangle(list(new_geo.total_bounds))
    
    # Get dates for rainfall a month before flood
    base = pd.to_datetime(flood_start)
    date_list = [(base - datetime.timedelta(days=x)).strftime('%Y-%m-%d') for x in range(31)]
    date_list = list(reversed(date_list))
    dates = [(date_list[num-1], date_list[num]) for num in range(1, 31)]
    
    # Create a list of all images filtered by date
    trmm_files = [trmm.filterDate(dates[i][0], dates[i][1]) for i in range(30)]
    
    # Assign export tasks for GEE
    task_list = []
    count = -30
    for trmm in trmm_files:
        # Filter by are of interest and return the mean for each image
        trmm_aoi = trmm.filterBounds(mwi_aoi)
        total = trmm_aoi.reduce(ee.Reducer.sum())
        task = ee.batch.Export.image.toDrive(image=total,
                                              region=mwi_aoi.getInfo()['coordinates'],
                                              description='13-mar',
                                              folder=folder,
                                              fileNamePrefix='total_precip_t' + str(count),
                                              maxPixels=1e12,
                                              scale=30,
                                              crs='EPSG:4326')
        task_list.append(task)
        count += 1
    for i in task_list:
        i.start()
        
    return i.status()



def rainfall_images(new_geo, flood_start, folder):
    """ Export GEE images for rainfall.
    Example: flood_start = '2019-03-13' for mozambique.
    """
    
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")# new_shapefile = "mwi_3-18.shp"
    
    chirps = chirps.select(['precipitation'])

    # setting the Area of Interest (AOI)
    mwi_aoi = ee.Geometry.Rectangle(list(new_geo.total_bounds))
    
    # Get dates for rainfall a month before flood
    base = pd.to_datetime(flood_start)
    date_list = [(base - datetime.timedelta(days=x)).strftime('%Y-%m-%d') for x in range(31)]
    date_list = list(reversed(date_list))
    dates = [(date_list[num-1], date_list[num]) for num in range(1, 31)]
    
    # Create a list of all images filtered by date
    chirps_files = [chirps.filterDate(dates[i][0], dates[i][1]) for i in range(30)]
    
    # Assign export tasks for GEE
    task_list = []
    count = -30
    for chirps in chirps_files:
        # Filter by are of interest and return the mean for each image
        chirps_aoi = chirps.filterBounds(mwi_aoi)
        total = chirps_aoi.reduce(ee.Reducer.sum())
        task = ee.batch.Export.image.toDrive(image=total,
                                              region=mwi_aoi.getInfo()['coordinates'],
                                              description='13-mar',
                                              folder=folder,
                                              fileNamePrefix='total_precip_t' + str(count),
                                              maxPixels=1e12,
                                              scale=30,
                                              crs='EPSG:4326')
        task_list.append(task)
        count += 1
    for i in task_list:
        i.start()
        
    return i.status()



rainfall_images(moz_geo, flood_start='2019-03-15', folder='moz_floods_2019-03-13')
rainfall_images(mwi_geo, flood_start='2019-03-15', folder='mwi_floods_2019-03-13')
rainfall_images(ken_geo, flood_start='2020-05-08', folder='mwi_floods_2020-05-06')


def rainfall_max_images(new_geo, flood_start, folder):
    """ Export GEE images for rainfall.
    Example: flood_start = '2019-03-13' for mozambique.
    """
    
    trmm = ee.ImageCollection("TRMM/3B42")# new_shapefile = "mwi_3-18.shp"
    
    trmm = trmm.select(['precipitation'])

    # setting the Area of Interest (AOI)
    mwi_aoi = ee.Geometry.Rectangle(list(new_geo.total_bounds))
    
    # Get dates for rainfall a month before flood
    base = pd.to_datetime(flood_start)
    date_list = [(base - datetime.timedelta(days=x)).strftime('%Y-%m-%d') for x in range(31)]
    date_list = list(reversed(date_list))
    dates = [(date_list[num-1], date_list[num]) for num in range(1, 31)]
    
    # Create a list of all images filtered by date
    trmm_files = [trmm.filterDate(dates[i][0], dates[i][1]) for i in range(30)]
    
    # Assign export tasks for GEE
    task_list = []
    count = -30
    for trmm in trmm_files:
        # Filter by are of interest and return the mean for each image
        trmm_aoi = trmm.filterBounds(mwi_aoi)
        maximum = trmm_aoi.reduce(ee.Reducer.max())
        task = ee.batch.Export.image.toDrive(image=maximum,
                                              region=mwi_aoi.getInfo()['coordinates'],
                                              description='13-mar',
                                              folder=folder,
                                              fileNamePrefix='max_precip_t' + str(count),
                                              maxPixels=1e12,
                                              scale=30,
                                              crs='EPSG:4326')
        task_list.append(task)
        count += 1
    for i in task_list:
        i.start()
        
    return i.status()
        

def rainfall_max_images(new_geo, flood_start, folder):
    """ Export GEE images for rainfall.
    Example: flood_start = '2019-03-13' for mozambique.
    """
    
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")# new_shapefile = "mwi_3-18.shp"
    
    chirps = chirps.select(['precipitation'])

    # setting the Area of Interest (AOI)
    mwi_aoi = ee.Geometry.Rectangle(list(new_geo.total_bounds))
    
    # Get dates for rainfall a month before flood
    base = pd.to_datetime(flood_start)
    date_list = [(base - datetime.timedelta(days=x)).strftime('%Y-%m-%d') for x in range(31)]
    date_list = list(reversed(date_list))
    dates = [(date_list[num-1], date_list[num]) for num in range(1, 31)]
    
    # Create a list of all images filtered by date
    chirps_files = [chirps.filterDate(dates[i][0], dates[i][1]) for i in range(30)]
    
    # Assign export tasks for GEE
    task_list = []
    count = -30
    for chirps in chirps_files:
        # Filter by are of interest and return the mean for each image
        chirps_aoi = chirps_files[20].filterBounds(mwi_aoi)
        total = chirps_aoi.reduce(ee.Reducer.max())
        task = ee.batch.Export.image.toDrive(image=total,
                                              region=mwi_aoi.getInfo()['coordinates'],
                                              description='13-mar',
                                              folder=folder,
                                              fileNamePrefix='max_precip_t' + str(count),
                                              maxPixels=1e12,
                                              scale=30,
                                              crs='EPSG:4326')
        task_list.append(task)
        count += 1
    for i in task_list:
        i.start()
        
    return i.status()
        

rainfall_max_images(moz_geo, flood_start='2019-03-15', folder='moz_floods_2019-03-13')
rainfall_max_images(mwi_geo, flood_start='2019-03-15', folder='mwi_floods_2019-03-13')
rainfall_max_images(ken_geo, flood_start='2020-05-08', folder='mwi_floods_2020-05-06')

def rainfall_std_images(new_geo, flood_start, folder):
    """ Export GEE images for rainfall.
    Example: flood_start = '2019-03-13' for mozambique.
    """
    
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")# new_shapefile = "mwi_3-18.shp"
    
    chirps = chirps.select(['precipitation'])

    # setting the Area of Interest (AOI)
    mwi_aoi = ee.Geometry.Rectangle(list(new_geo.total_bounds))
    
    # Get dates for rainfall a month before flood
    base = pd.to_datetime(flood_start)
    date_list = [(base - datetime.timedelta(days=x)).strftime('%Y-%m-%d') for x in range(31)]
    date_list = list(reversed(date_list))
    dates = [(date_list[num-1], date_list[num]) for num in range(1, 31)]
    
    # Create a list of all images filtered by date
    chirps_files = [chirps.filterDate(dates[i][0], dates[i][1]) for i in range(30)]
    
    # Assign export tasks for GEE
    task_list = []
    count = -30
    for chirps in chirps_files:
        # Filter by are of interest and return the mean for each image
        chirps_aoi = chirps.filterBounds(mwi_aoi)
        total = chirps_aoi.reduce(ee.Reducer.sum())
        task = ee.batch.Export.image.toDrive(image=total,
                                              region=mwi_aoi.getInfo()['coordinates'],
                                              description='13-mar',
                                              folder=folder,
                                              fileNamePrefix='std_precip_t' + str(count),
                                              maxPixels=1e12,
                                              scale=30,
                                              crs='EPSG:4326')
        task_list.append(task)
        count += 1
    for i in task_list:
        i.start()
        
    return i.status()
        
    return i.status()
        

rainfall_std_images(moz_geo, flood_start='2019-03-15', folder='moz_floods_2019-03-13')
rainfall_std_images(mwi_geo, flood_start='2019-03-15', folder='mwi_floods_2019-03-13')
rainfall_std_images(ken_geo, flood_start='2020-05-08', folder='mwi_floods_2020-05-06')


def smap_images(new_geo, flood_start, folder):
    """ Export GEE images for rainfall.
    Example: flood_start = '2019-03-13' for mwimabique.
    """
    
    ssm = ee.ImageCollection("NASA_USDA/HSL/SMAP_soil_moisture") # new_shapefile = "mwi_3-18.shp"
    
    ssm = ssm.select(['ssm'])

    # setting the Area of Interest (AOI)
    mwi_aoi = ee.Geometry.Rectangle(list(new_geo.total_bounds))
    
    # Get dates for rainfall a month before flood
    base = pd.to_datetime(flood_start)
    date_list = [(base - datetime.timedelta(days=x)).strftime('%Y-%m-%d') for x in range(31)]
    date_list = list(reversed(date_list))
    dates = [(date_list[num-1], date_list[num]) for num in range(1, 31)]
    
    # Create a list of all images filtered by date
    ssm_files = [ssm.filterDate(dates[i][0], dates[i][1]) for i in range(30)]
    
    # Assign export tasks for GEE
    task_list = []
    count = -30
    for ssm in ssm_files:
        # Filter by are of interest and return the mean for each image
        ssm_aoi = ssm.filterBounds(mwi_aoi)
        total = ssm_aoi.reduce(ee.Reducer.sum())
        task = ee.batch.Export.image.toDrive(image=total,
                                              region=mwi_aoi.getInfo()['coordinates'],
                                              description='13-mar',
                                              folder=folder,
                                              fileNamePrefix='ssm_t' + str(count),
                                              maxPixels=1e12,
                                              scale=30,
                                              crs='EPSG:4326')
        task_list.append(task)
        count += 1        
    for i in task_list:
        i.start()
        
    return i.status()
        

smap_images(moz_geo, flood_start='2019-03-15', folder='moz_floods_2019-03-13')
smap_images(mwi_geo, flood_start='2019-03-15', folder='mwi_floods_2019-03-13')
smap_images(ken_geo, flood_start='2020-05-08', folder='mwi_floods_2020-05-06')


def lc_images(new_geo, flood_start, folder)  :
        
    modis = ee.ImageCollection("MODIS/006/MCD12Q1") # new_shapefile = "mwi_3-18.shp"
    modis = modis.select(['LC_Type1'])
    
    # setting the Area of Interest (AOI)
    mwi_aoi = ee.Geometry.Rectangle(list(new_geo.total_bounds))
    modis_aoi = modis.filterBounds(mwi_aoi)
    mode = modis_aoi.reduce(ee.Reducer.mode())
    #least_cloudy = ee.Image(modis_aoi.sort('CLOUD_COVER').first())
    task = ee.batch.Export.image.toDrive(image=mode,
                                          region=mwi_aoi.getInfo()['coordinates'],
                                          description='13-mar',
                                          folder=folder,
                                          fileNamePrefix='land_cover_mode',
                                          scale=30,
                                          crs='EPSG:4326')
    task.start()
    return task.status()
        
  

lc_images(moz_geo, flood_start='2019-03-15', folder='moz_floods_2019-03-13')
lc_images(mwi_geo, flood_start='2019-03-15', folder='mwi_floods_2019-03-13')
lc_images(ken_geo, flood_start='2020-05-08', folder='mwi_floods_2020-05-06')
    
    
def elevation_images(new_geo, flood_start, folder):
    srtm = ee.Image("USGS/SRTMGL1_003") 
    srtm = srtm.select('elevation')
    mwi_aoi = ee.Geometry.Rectangle(list(new_geo.total_bounds))
    srtm_aoi = srtm.clip(mwi_aoi)    
    task = ee.batch.Export.image.toDrive(image=srtm_aoi,
                                          region=mwi_aoi.getInfo()['coordinates'],
                                          description='13-mar',
                                          folder=folder,
                                          fileNamePrefix='elevation',
                                          scale=30,
                                          crs='EPSG:4326')
    task.start()
    return task.status()

elevation_images(moz_geo, flood_start='2019-03-15', folder='moz_floods_2019-03-13')
elevation_images(mwi_geo, flood_start='2019-03-15', folder='mwi_floods_2019-03-13')
elevation_images(ken_geo, flood_start='2020-05-08', folder='mwi_floods_2020-05-06')

    
def slope_images(new_geo, flood_start, folder):        
    srtm = ee.Image("USGS/SRTMGL1_003") 
    srtm = srtm.select('elevation')
    slope = ee.Terrain.slope(srtm)
    mwi_aoi = ee.Geometry.Rectangle(list(new_geo.total_bounds))
    srtm_aoi = slope.clip(mwi_aoi)           
    task = ee.batch.Export.image.toDrive(image=srtm_aoi,
                                          region=mwi_aoi.getInfo()['coordinates'],
                                          description='13-mar',
                                          folder=folder,
                                          fileNamePrefix='slope',
                                          scale=30,
                                          crs='EPSG:4326')
    task.start()
    return task.status()


slope_images(moz_geo, flood_start='2019-03-15', folder='moz_floods_2019-03-13')
slope_images(mwi_geo, flood_start='2019-03-15', folder='mwi_floods_2019-03-13')
slope_images(ken_geo, flood_start='2020-05-08', folder='mwi_floods_2020-05-06')



def ndvi_images(new_geo, flood_start, folder):    
    """ Export GEE images for rainfall.
    Example: flood_start = '2019-03-13' for mwimabique.
    """    
    ndvi = ee.ImageCollection('MODIS/006/MOD13A2')
    # choose dates
    ndvi = ndvi.select('NDVI')
    # setting the Area of Interest (AOI)
    # setting the Area of Interest (AOI)
    mwi_aoi = ee.Geometry.Rectangle(list(new_geo.total_bounds))
    
    # Get dates for rainfall a month before flood
    base = pd.to_datetime(flood_start)
    date_list = [(base - datetime.timedelta(days=x)).strftime('%Y-%m-%d') for x in range(31)]
    date_list = list(reversed(date_list))
    dates = [(date_list[num-1], date_list[num]) for num in range(1, 31)]
    
    # Create a list of all images filtered by date
    ndvi_files = [ndvi.filterDate(dates[i][0], dates[i][1]) for i in range(30)]
    # the least cloudy image
    
    # Assign export tasks for GEE
    task_list = []
    count = -30
    for ndvi in ndvi_files:
        # Filter by are of interest and return the mean for each image
        ndvi_aoi = ndvi.filterBounds(mwi_aoi)
        total = ndvi_aoi.reduce(ee.Reducer.sum())
        task = ee.batch.Export.image.toDrive(image=total.float(),
                                              region=mwi_aoi.getInfo()['coordinates'],
                                              description='13-mar',
                                              folder=folder,
                                              fileNamePrefix='ndvi_t' + str(count),
                                              maxPixels=1e12,
                                              scale=30,
                                              crs='EPSG:4326')
        task_list.append(task)
        count += 1        
    for i in task_list:
        i.start()
        
    return i.status()
    


ndvi_images(moz_geo, flood_start='2019-03-15', folder='moz_floods_2019-03-13')
ndvi_images(mwi_geo, flood_start='2019-03-15', folder='mwi_floods_2019-03-13')
ndvi_images(ken_geo, flood_start='2020-05-08', folder='mwi_floods_2020-05-06')


#### Get soil organic carbon
from tiff_image import CRS

moz_crs = CRS("data/moz_soil_organic_carbon.tif")
moz_crs.convert("data/floods_13-03-2020/moz_feats/soc_mean.tif")

mwi_crs = CRS("data/mwi_soil_organic_carbon.tif")
mwi_crs.convert("data/floods_13-03-2020/mwi_feats/soc_mean.tif")















  
