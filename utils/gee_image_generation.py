#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ee
from datetime import datetime as dt

ee.Initialize()

"""
This script contains functionality for exporting Google Earth Engine images to
Google Drive folder.
"""


##############   Training flood data     ######################################

##### == Check if these exports match in the provided datatset == 


trmm = ee.ImageCollection("TRMM/3B42")# new_shapefile = "mwi_3-18.shp"

trmm = trmm.select(['precipitation', 'HQprecipitation', 'IRprecipitation'])


trmm_1 = trmm.filterDate('2015-03-08','2015-03-15')

trmm_2 = trmm.filterDate('2015-03-01','2015-03-08')

trmm_3 = trmm.filterDate('2015-02-22','2015-03-01')

trmm_4 = trmm.filterDate('2015-02-15','2015-02-22')

trmm_5 = trmm.filterDate('2015-02-08','2015-02-22')

trmm_6 = trmm.filterDate('2015-02-01','2015-02-08')

trmm_7 = trmm.filterDate('2015-01-25','2015-02-01')

trmm_8 = trmm.filterDate('2015-01-18','2015-01-25')

trmm_9 = trmm.filterDate('2015-01-11','2015-01-18')

trmm_10 = trmm.filterDate('2015-01-04','2015-01-11')

trmm_11 = trmm.filterDate('2014-12-28','2015-01-04')

trmm_12 = trmm.filterDate('2014-12-21','2014-12-28')

trmm_13 = trmm.filterDate('2014-12-14','2014-12-21')

trmm_14 = trmm.filterDate('2014-12-07','2014-12-14')

trmm_15 = trmm.filterDate('2014-11-30','2014-12-07')

trmm_16 = trmm.filterDate('2014-11-23','2014-11-30')

trmm_17 = trmm.filterDate('2014-11-16','2014-11-23')

# setting the Area of Interest (AOI)
mwi_aoi = ee.Geometry.Rectangle([34.255, -16.645,  35.865, -15.205])

#trmm_2_aoi = trmm_2.filterBounds(mwi_aoi)
#std = trmm_2.reduce(ee.Reducer.std())

trmm_list = [trmm_1, trmm_2, trmm_3, trmm_4, trmm_5, trmm_6, trmm_7, trmm_8, trmm_9,
              trmm_10, trmm_11, trmm_12, trmm_13, trmm_14, trmm_15, trmm_16, trmm_17]

task_list = []
count = 1
for trmm in trmm_list:
    trmm_aoi = trmm.filterBounds(mwi_aoi)
    mean = trmm_aoi.reduce(ee.Reducer.mean())
   
    task = ee.batch.Export.image.toDrive(image=mean,
                                          region=mwi_aoi.getInfo()['coordinates'],
                                          description='mwi_precip_for_15-03-15_flood',
                                          folder='mwi_floods',
                                          fileNamePrefix='precip_wk' + str(count) + '-tr',
                                          scale=30,
                                          crs='EPSG:4326')
    task_list.append(task)
    count += 1
    
    
for i in task_list:
    i.start()



task_list = []
count = 1
for trmm in trmm_list:
    trmm_aoi = trmm.filterBounds(mwi_aoi)
    total = trmm_aoi.reduce(ee.Reducer.sum())
   
    task = ee.batch.Export.image.toDrive(image=total,
                                          region=mwi_aoi.getInfo()['coordinates'],
                                          description='mwi_precip_for_15-03-15_flood',
                                          folder='mwi_floods',
                                          fileNamePrefix='precip_wk' + str(count) + '_total-tr',
                                          scale=30,
                                          crs='EPSG:4326')
    task_list.append(task)
    count += 1
    
for i in task_list:
    i.start()

task_list = []
count = 1
for trmm in trmm_list:
    trmm_aoi = trmm.filterBounds(mwi_aoi)
    var = trmm_aoi.reduce(ee.Reducer.variance())
   
    task = ee.batch.Export.image.toDrive(image=var,
                                          region=mwi_aoi.getInfo()['coordinates'],
                                          description='mwi_precip_for_15-03-15_flood',
                                          folder='mwi_floods',
                                          fileNamePrefix='precip_wk' + str(count) + '_var-tr',
                                          scale=30,
                                          crs='EPSG:4326')
    task_list.append(task)
    count += 1
    
for i in task_list:
    i.start()
    
    
smap = ee.ImageCollection("NASA_USDA/HSL/SMAP_soil_moisture")
# choose dates
smap = smap.filterDate('2014-12-04', '2015-01-04')
# Choose band
smap = smap.select(['ssm', 'susm'])

aoi_bound = smap.filterBounds(mwi_aoi)

# the least cloudy image
least_cloudy = ee.Image(aoi_bound.sort('CLOUD_COVER').first())
# how cloudy is it?
#print('Cloud Cover (%):', least_cloudy.get('CLOUD_COVER').getInfo())        
# when was this image taken?
# date = ee.Date(least_cloudy.get('system:time_start'))
# time = date.getInfo()['value']/1000.
# dt.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
task = ee.batch.Export.image.toDrive(image=least_cloudy,
                                      region=mwi_aoi.getInfo()['coordinates'],
                                      description='train_smap',
                                      folder='train',
                                      fileNamePrefix='train_smap',
                                      scale=30,
                                      crs='EPSG:4326')
task.start()


## Ndvi





ndvi = ee.ImageCollection('MODIS/006/MOD13A2')
# choose dates
ndvi = ndvi.filterDate('2014-12-04', '2015-01-04')
ndvi = ndvi.select(['NDVI', 'EVI'])

# setting the Area of Interest (AOI)
#moz_aoi = ee.Geometry.Rectangle(aoi)
aoi_bound = ndvi.filterBounds(mwi_aoi)

# the least cloudy image
least_cloudy = ee.Image(aoi_bound.sort('CLOUD_COVER').first())
# how cloudy is it?
print('Cloud Cover (%):', least_cloudy.get('CLOUD_COVER').getInfo())        
# when was this image taken?
date = ee.Date(least_cloudy.get('system:time_start'))
time = date.getInfo()['value']/1000.
dt.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
task = ee.batch.Export.image.toDrive(image=least_cloudy,
                                      region=mwi_aoi.getInfo()['coordinates'],
                                      description='train' + '_ndvi',
                                      folder='train',
                                      fileNamePrefix='ndvi',
                                      scale=30,
                                      crs='EPSG:4326')
task.start()



## Evapotranspiration

evap = ee.ImageCollection("MODIS/006/MOD16A2")

# choose dates
evap = evap.filterDate('2014-12-04', '2015-01-04')
evap = evap.select('ET')

# setting the Area of Interest (AOI)
aoi_bound = evap.filterBounds(mwi_aoi)

# the least cloudy image
least_cloudy = ee.Image(aoi_bound.sort('CLOUD_COVER').first())
# how cloudy is it?
print('Cloud Cover (%):', least_cloudy.get('CLOUD_COVER').getInfo())        
# when was this image taken?
date = ee.Date(least_cloudy.get('system:time_start'))
time = date.getInfo()['value']/1000.
dt.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
task = ee.batch.Export.image.toDrive(image=least_cloudy,
                                      region=mwi_aoi.getInfo()['coordinates'],
                                      description='train' + '_evap',
                                      folder='train',
                                      fileNamePrefix='evap',
                                      scale=30,
                                      crs='EPSG:4326')
task.start()





## climate fetaures

clim = ee.ImageCollection("ECMWF/ERA5/DAILY")

# choose dates
clim = clim.filterDate('2015-01-12','2015-01-14')
clim = clim.select(['mean_2m_air_temperature', 'dewpoint_2m_temperature', 
                    'total_precipitation', 'u_component_of_wind_10m', 
                    'v_component_of_wind_10m'])

# setting the Area of Interest (AOI)
aoi_bound = clim.filterBounds(mwi_aoi)

# the least cloudy image
least_cloudy = ee.Image(aoi_bound.sort('CLOUD_COVER').first())
# how cloudy is it?
print('Cloud Cover (%):', least_cloudy.get('CLOUD_COVER').getInfo())        
# when was this image taken?
date = ee.Date(least_cloudy.get('system:time_start'))
time = date.getInfo()['value']/1000.
dt.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
task = ee.batch.Export.image.toDrive(image=least_cloudy,
                                      region=mwi_aoi.getInfo()['coordinates'],
                                      description='train' + '_clim',
                                      folder='train',
                                      fileNamePrefix='clim',
                                      scale=30,
                                      crs='EPSG:4326')
task.start()




## Land Surface Temperature

land_temp = ee.ImageCollection("MODIS/006/MOD11A1")


lt_1 = land_temp.filterDate('2015-03-08','2015-03-15')

lt_2 = land_temp.filterDate('2015-03-01','2015-03-08')

lt_3 = land_temp.filterDate('2015-02-22','2015-03-01')

lt_4 = land_temp.filterDate('2015-02-15','2015-02-22')

lt_5 = land_temp.filterDate('2015-02-08','2015-02-22')

lt_6 = land_temp.filterDate('2015-02-01','2015-02-08')

lt_7 = land_temp.filterDate('2015-01-25','2015-02-01')

lt_8 = land_temp.filterDate('2015-01-18','2015-01-25')

lt_9 = land_temp.filterDate('2015-01-11','2015-01-18')

lt_10 = land_temp.filterDate('2015-01-04','2015-01-11')

lt_11 = land_temp.filterDate('2014-12-28','2015-01-04')

lt_12 = land_temp.filterDate('2014-12-21','2014-12-28')

lt_13 = land_temp.filterDate('2014-12-14','2014-12-21')

lt_14 = land_temp.filterDate('2014-12-07','2014-12-14')

lt_15 = land_temp.filterDate('2014-11-30','2014-12-07')

lt_16 = land_temp.filterDate('2014-11-23','2014-11-30')

lt_17 = land_temp.filterDate('2014-11-16','2014-11-23')



lt_list = [lt_1, lt_2, lt_3, lt_4, lt_5, lt_6, lt_7, lt_8, lt_9,
              lt_10, lt_11, lt_12, lt_13, lt_14, lt_15, lt_16, lt_17]

task_list = []
count = 1
for lt in lt_list:
    lt_aoi = lt.select('LST_Day_1km').filterBounds(mwi_aoi)
    mean = lt_aoi.reduce(ee.Reducer.mean())
   
    task = ee.batch.Export.image.toDrive(image=mean,
                                          region=mwi_aoi.getInfo()['coordinates'],
                                          description='land_temp_for_15-03-15_flood',
                                          folder='mwi_land_temp',
                                          fileNamePrefix='land_temp_wk' + str(count) + '-tr',
                                          scale=30,
                                          crs='EPSG:4326')
    task_list.append(task)
    count += 1
    
    
for i in task_list:
    i.start()





#############    Test  flood data   ###########################################

##### == Check if these exports match in the provided test set == 


trmm = ee.ImageCollection("TRMM/3B42")# new_shapefile = "mwi_3-18.shp"

trmm = trmm.select(['precipitation', 'HQprecipitation', 'IRprecipitation'])


trmm_1 = trmm.filterDate('2019-05-12','2019-05-19')

trmm_2 = trmm.filterDate('2019-05-05','2019-05-12')

trmm_3 = trmm.filterDate('2019-04-28','2019-05-05')

trmm_4 = trmm.filterDate('2019-04-21','2019-04-28')

trmm_5 = trmm.filterDate('2019-04-14','2019-04-21')

trmm_6 = trmm.filterDate('2019-04-07','2019-04-14')

trmm_7 = trmm.filterDate('2019-03-31','2019-04-07')

trmm_8 = trmm.filterDate('2019-03-24','2019-03-31')

trmm_9 = trmm.filterDate('2019-03-17','2019-03-24')

trmm_10 = trmm.filterDate('2019-03-10','2019-03-17')

trmm_11 = trmm.filterDate('2019-03-03','2019-03-10')

trmm_12 = trmm.filterDate('2019-02-24','2019-03-03')

trmm_13 = trmm.filterDate('2019-02-17','2019-02-24')

trmm_14 = trmm.filterDate('2019-02-10','2019-02-17')

trmm_15 = trmm.filterDate('2019-02-03','2019-02-10')

trmm_16 = trmm.filterDate('2019-01-27','2019-02-03')

trmm_17 = trmm.filterDate('2019-01-20','2019-01-27')


# setting the Area of Interest (AOI)
mwi_aoi = ee.Geometry.Rectangle([34.255, -16.645,  35.865, -15.205])

#trmm_2_aoi = trmm_2.filterBounds(mwi_aoi)
#std = trmm_2.reduce(ee.Reducer.std())

trmm_list = [trmm_1, trmm_2, trmm_3, trmm_4, trmm_5, trmm_6, trmm_7, trmm_8, trmm_9,
              trmm_10, trmm_11, trmm_12, trmm_13, trmm_14, trmm_15, trmm_16, trmm_17]
task_list = []
count = 1
for trmm in trmm_list:
    trmm_aoi = trmm.filterBounds(mwi_aoi)
    total = trmm_aoi.reduce(ee.Reducer.sum())
   
    task = ee.batch.Export.image.toDrive(image=total,
                                          region=mwi_aoi.getInfo()['coordinates'],
                                          description='mwi_precip_for_05-19_flood',
                                          folder='my_gdrive_folder',
                                          fileNamePrefix='precip_wk' + str(count) + '_total-te',
                                          scale=30,
                                          crs='EPSG:4326')
    task_list.append(task)
    count += 1
    
for i in task_list:
    i.start()


    
smap = ee.ImageCollection("NASA_USDA/HSL/SMAP_soil_moisture")
# choose dates
smap = smap.filterDate('2019-02-10', '2019-03-10')
# Choose band
smap = smap.select(['ssm', 'susm'])

aoi_bound = smap.filterBounds(mwi_aoi)

# the least cloudy image
least_cloudy = ee.Image(aoi_bound.sort('CLOUD_COVER').first())
# how cloudy is it?
#print('Cloud Cover (%):', least_cloudy.get('CLOUD_COVER').getInfo())        
# when was this image taken?
# date = ee.Date(least_cloudy.get('system:time_start'))
# time = date.getInfo()['value']/1000.
# dt.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
task = ee.batch.Export.image.toDrive(image=least_cloudy,
                                      region=mwi_aoi.getInfo()['coordinates'],
                                      description='test_smap',
                                      folder='test',
                                      fileNamePrefix='test_smap',
                                      scale=30,
                                      crs='EPSG:4326')
task.start()


## ndvi

ndvi = ee.ImageCollection('MODIS/006/MOD13A2')
# choose dates
ndvi = ndvi.filterDate('2019-02-10', '2019-03-10')
ndvi = ndvi.select(['NDVI', 'EVI'])

# setting the Area of Interest (AOI)
#moz_aoi = ee.Geometry.Rectangle(aoi)
aoi_bound = ndvi.filterBounds(mwi_aoi)

# the least cloudy image
least_cloudy = ee.Image(aoi_bound.sort('CLOUD_COVER').first())
# how cloudy is it?
print('Cloud Cover (%):', least_cloudy.get('CLOUD_COVER').getInfo())        
# when was this image taken?
date = ee.Date(least_cloudy.get('system:time_start'))
time = date.getInfo()['value']/1000.
dt.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
task = ee.batch.Export.image.toDrive(image=least_cloudy,
                                      region=mwi_aoi.getInfo()['coordinates'],
                                      description='test' + '_ndvi',
                                      folder='test',
                                      fileNamePrefix='ndvi',
                                      scale=30,
                                      crs='EPSG:4326')
task.start()



## Evapotranspiration

evap = ee.ImageCollection("MODIS/006/MOD16A2")

# choose dates
evap = evap.filterDate('2019-02-10', '2019-03-10')
evap = evap.select('ET')

# setting the Area of Interest (AOI)
aoi_bound = evap.filterBounds(mwi_aoi)

# the least cloudy image
least_cloudy = ee.Image(aoi_bound.sort('CLOUD_COVER').first())
# how cloudy is it?
print('Cloud Cover (%):', least_cloudy.get('CLOUD_COVER').getInfo())        
# when was this image taken?
date = ee.Date(least_cloudy.get('system:time_start'))
time = date.getInfo()['value']/1000.
dt.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
task = ee.batch.Export.image.toDrive(image=least_cloudy,
                                      region=mwi_aoi.getInfo()['coordinates'],
                                      description='test' + '_evap',
                                      folder='test',
                                      fileNamePrefix='evap',
                                      scale=30,
                                      crs='EPSG:4326')
task.start()


## climate fetaures

clim = ee.ImageCollection("ECMWF/ERA5/DAILY")

# choose dates
clim = clim.filterDate('2019-03-13','2019-03-15')
clim = clim.select(['mean_2m_air_temperature', 'dewpoint_2m_temperature', 
                    'total_precipitation', 'u_component_of_wind_10m', 
                    'v_component_of_wind_10m'])

# setting the Area of Interest (AOI)
aoi_bound = clim.filterBounds(mwi_aoi)

# the least cloudy image
least_cloudy = ee.Image(aoi_bound.sort('CLOUD_COVER').first())
# how cloudy is it?
print('Cloud Cover (%):', least_cloudy.get('CLOUD_COVER').getInfo())        
# when was this image taken?
date = ee.Date(least_cloudy.get('system:time_start'))
time = date.getInfo()['value']/1000.
dt.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
task = ee.batch.Export.image.toDrive(image=least_cloudy,
                                      region=mwi_aoi.getInfo()['coordinates'],
                                      description='test' + '_clim',
                                      folder='test',
                                      fileNamePrefix='clim',
                                      scale=30,
                                      crs='EPSG:4326')
task.start()






## Land Surface Temperature

land_temp = ee.ImageCollection("MODIS/006/MOD11A1")



lt_1 = lt.filterDate('2019-05-12','2019-05-19')

lt_2 = lt.filterDate('2019-05-05','2019-05-12')

lt_3 = lt.filterDate('2019-04-28','2019-05-05')

lt_4 = lt.filterDate('2019-04-21','2019-04-28')

lt_5 = lt.filterDate('2019-04-14','2019-04-21')

lt_6 = lt.filterDate('2019-04-07','2019-04-14')

lt_7 = lt.filterDate('2019-03-31','2019-04-07')

lt_8 = lt.filterDate('2019-03-24','2019-03-31')

lt_9 = lt.filterDate('2019-03-17','2019-03-24')

lt_10 = lt.filterDate('2019-03-10','2019-03-17')

lt_11 = lt.filterDate('2019-03-03','2019-03-10')

lt_12 = lt.filterDate('2019-02-24','2019-03-03')

lt_13 = lt.filterDate('2019-02-17','2019-02-24')

lt_14 = lt.filterDate('2019-02-10','2019-02-17')

lt_15 = lt.filterDate('2019-02-03','2019-02-10')

lt_16 = lt.filterDate('2019-01-27','2019-02-03')

lt_17 = lt.filterDate('2019-01-20','2019-01-27')



lt_list = [lt_1, lt_2, lt_3, lt_4, lt_5, lt_6, lt_7, lt_8, lt_9,
              lt_10, lt_11, lt_12, lt_13, lt_14, lt_15, lt_16, lt_17]

task_list = []
count = 1
for lt in lt_list:
    lt_aoi = lt.select('LST_Day_1km').filterBounds(mwi_aoi)
    mean = lt_aoi.reduce(ee.Reducer.mean())
   
    task = ee.batch.Export.image.toDrive(image=mean,
                                          region=mwi_aoi.getInfo()['coordinates'],
                                          description='land_temp_for_05-19_flood',
                                          folder='mwi_land_temp',
                                          fileNamePrefix='land_temp_wk' + str(count) + '-te',
                                          scale=30,
                                          crs='EPSG:4326')
    task_list.append(task)
    count += 1
    
    
for i in task_list:
    i.start()



###############################################################################
###############################################################################

# Extract new features for the new flood event data 

## Functions


def begin_trmm_export(aoi, date_list, stat_name, folder):
    """ Get precip tif files from GEE """   
    if stat_name == 'mean':
        stat = ee.Reducer.mean()
    if stat_name == 'total':
        stat = ee.Reducer.sum()
    if stat_name == 'var':
        stat = ee.Reducer.variance()          
    task_list = []
    count = 1
    for trmm in trmm_list:
        trmm_aoi = trmm.filterBounds(aoi)
        task = ee.batch.Export.image.toDrive(image=trmm_aoi.reduce(stat)       ,
                                              region=mwi_aoi.getInfo()['coordinates'],
                                              description=folder + '_' + stat_name,
                                              folder=folder,
                                              fileNamePrefix='precip_wk-' + str(count) + f'_{stat_name}',
                                              scale=30,
                                              crs='EPSG:4326')
        task_list.append(task)
        count += 1    
    for i in task_list:
        print(f"Start task {i}")
        i.start()
        
    return task.status()



def begin_lc_export(aoi, start_date, end_date, folder):
    """ Get Landcover data. """
    modis = ee.ImageCollection('MODIS/006/MCD12Q1')
    # choose dates
    modis = modis.filterDate(start_date,end_date)

    # setting the Area of Interest (AOI)
    #moz_aoi = ee.Geometry.Rectangle(aoi)
    aoi_bound = modis.filterBounds(aoi)
    
    # the least cloudy image
    least_cloudy = ee.Image(aoi_bound.sort('CLOUD_COVER').first())
    # how cloudy is it?
    print('Cloud Cover (%):', least_cloudy.get('CLOUD_COVER').getInfo())        
    # when was this image taken?
    date = ee.Date(least_cloudy.get('system:time_start'))
    time = date.getInfo()['value']/1000.
    dt.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
    task = ee.batch.Export.image.toDrive(image=least_cloudy,
                                          region=aoi.getInfo()['coordinates'],
                                          description=folder + '_lc',
                                          folder=folder,
                                          fileNamePrefix='lc_type1',
                                          scale=30,
                                          crs='EPSG:4326')
    task.start()
    return task.status()



def begin_smap_export(aoi, start_date, end_date, folder, dataset=['ssm', 'susm']):
    """ Get Soil Moisture. """
    smap = ee.ImageCollection("NASA_USDA/HSL/SMAP_soil_moisture")
    # choose dates
    smap = smap.filterDate(start_date,end_date)
    # Choose band
    smap = smap.select(dataset)
    aoi_bound = smap.filterBounds(aoi)
    
    # the least cloudy image
    least_cloudy = ee.Image(aoi_bound.sort('CLOUD_COVER').first())
    # how cloudy is it?
    print('Cloud Cover (%):', least_cloudy.get('CLOUD_COVER').getInfo())        
    # when was this image taken?
    date = ee.Date(least_cloudy.get('system:time_start'))
    time = date.getInfo()['value']/1000.
    dt.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
    task = ee.batch.Export.image.toDrive(image=least_cloudy,
                                          region=aoi.getInfo()['coordinates'],
                                          description=folder + '_smap',
                                          folder=folder,
                                          fileNamePrefix='smap',
                                          maxPixels=1e9,
                                          scale=30,
                                          crs='EPSG:4326')
    task.start()
    return task.status()


def begin_ndvi_export(aoi, start_date, end_date, folder, dataset=['NDVI', 'EVI']):
    """ Get Normalized difference vegetation index/enhanced vegetation index."""
    ndvi = ee.ImageCollection('MODIS/006/MOD13A2')
    # choose dates
    ndvi = ndvi.filterDate(start_date, end_date)
    ndvi = ndvi.select(dataset)

    # setting the Area of Interest (AOI)
    #moz_aoi = ee.Geometry.Rectangle(aoi)
    aoi_bound = ndvi.filterBounds(aoi)

    # the least cloudy image
    least_cloudy = ee.Image(aoi_bound.sort('CLOUD_COVER').first())
    # how cloudy is it?
    print('Cloud Cover (%):', least_cloudy.get('CLOUD_COVER').getInfo())        
    # when was this image taken?
    date = ee.Date(least_cloudy.get('system:time_start'))
    time = date.getInfo()['value']/1000.
    dt.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
    task = ee.batch.Export.image.toDrive(image=least_cloudy,
                                          region=aoi.getInfo()['coordinates'],
                                          description=folder + '_ndvi',
                                          folder=folder,
                                          fileNamePrefix='ndvi',
                                          scale=30,
                                          crs='EPSG:4326')
    task.start()
    return task.status()


def begin_evap_export(aoi, start_date, end_date, folder, dataset=['ET']):
    """Get evapotranspiration. """        
    evap = ee.ImageCollection("MODIS/006/MOD16A2")
    # choose dates
    evap = evap.filterDate(start_date, end_date)
    evap = evap.select(dataset)
    # setting the Area of Interest (AOI)
    aoi_bound = evap.filterBounds(aoi)
    # the least cloudy image
    least_cloudy = ee.Image(aoi_bound.sort('CLOUD_COVER').first())
    # how cloudy is it?
    print('Cloud Cover (%):', least_cloudy.get('CLOUD_COVER').getInfo())        
    # when was this image taken?
    date = ee.Date(least_cloudy.get('system:time_start'))
    time = date.getInfo()['value']/1000.
    dt.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
    task = ee.batch.Export.image.toDrive(image=least_cloudy,
                                          region=aoi.getInfo()['coordinates'],
                                          description=folder + '_evap',
                                          folder=folder,
                                          fileNamePrefix='evap',
                                          scale=30,
                                          crs='EPSG:4326')
    task.start()
    return task.status()
    

def begin_clim_export(aoi, start_date, end_date, folder):
    """ Get climate features. """
    clim = ee.ImageCollection("ECMWF/ERA5/DAILY")  
    # choose dates
    clim = clim.filterDate(start_date, end_date)
    clim = clim.select(['mean_2m_air_temperature', 'dewpoint_2m_temperature', 
                        'total_precipitation', 'u_component_of_wind_10m', 
                        'v_component_of_wind_10m'])   
    # setting the Area of Interest (AOI)
    aoi_bound = clim.filterBounds(aoi)   
    # the least cloudy image
    least_cloudy = ee.Image(aoi_bound.sort('CLOUD_COVER').first())
    # how cloudy is it?
    print('Cloud Cover (%):', least_cloudy.get('CLOUD_COVER').getInfo())        
    # when was this image taken?
    date = ee.Date(least_cloudy.get('system:time_start'))
    time = date.getInfo()['value']/1000.
    dt.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
    task = ee.batch.Export.image.toDrive(image=least_cloudy,
                                          region=aoi.getInfo()['coordinates'],
                                          description=folder + '_clim',
                                          folder=folder,
                                          fileNamePrefix='clim',
                                          maxPixels=1e9,
                                          scale=30,
                                          crs='EPSG:4326')
    task.start()
    return task.status()


   
###############################################################################


######################   Malawi flood 03-18      #######################

trmm = ee.ImageCollection("TRMM/3B42")# new_shapefile = "mwi_3-18.shp"

trmm = trmm.select(['precipitation', 'HQprecipitation', 'IRprecipitation'])


trmm_1 = trmm.filterDate('2018-02-27','2018-03-04')

trmm_2 = trmm.filterDate('2018-02-20','2018-02-27')

trmm_3 = trmm.filterDate('2018-02-13','2018-02-20')

trmm_4 = trmm.filterDate('2018-02-05','2018-02-13')

trmm_5 = trmm.filterDate('2018-01-30','2018-02-05')

trmm_6 = trmm.filterDate('2018-01-23','2018-01-30')

trmm_7 = trmm.filterDate('2018-01-14','2018-01-23')

trmm_8 = trmm.filterDate('2018-01-07','2018-01-14')

trmm_9 = trmm.filterDate('2017-12-31','2018-01-07')

trmm_10 = trmm.filterDate('2017-12-24','2017-12-31')

trmm_11 = trmm.filterDate('2017-12-17','2017-12-24')

trmm_12 = trmm.filterDate('2017-12-10','2017-12-17')

trmm_13 = trmm.filterDate('2017-12-03','2017-12-10')

trmm_14 = trmm.filterDate('2017-11-26','2017-12-03')

trmm_15 = trmm.filterDate('2017-11-19','2017-11-26')

trmm_16 = trmm.filterDate('2017-11-12','2017-11-19')

trmm_17 = trmm.filterDate('2017-11-05','2017-11-12')


# setting the Area of Interest (AOI)
mwi_aoi = ee.Geometry.Rectangle([34.  , -17.01,  36.  , -15.01])


trmm_list = [trmm_1, trmm_2, trmm_3, trmm_4, trmm_5, trmm_6, trmm_7, trmm_8, trmm_9,
              trmm_10, trmm_11, trmm_12, trmm_13, trmm_14, trmm_15, trmm_16, trmm_17]


# Get precipitations
mwi_precip_mean = begin_trmm_export(mwi_aoi, trmm_list, 'mean', 'mwi_floods_03-18')
mwi_precip_total = begin_trmm_export(mwi_aoi, trmm_list, 'total', 'mwi_floods_03-18')
mwi_precip_var = begin_trmm_export(mwi_aoi, trmm_list, 'var', 'mwi_floods_03-18')


# Get landcover
mwi_lc = begin_lc_export(mwi_aoi, '2017-01-01', '2018-01-01', 'mwi_floods_03-18')

# Get smap
mwi_smap = begin_smap_export(mwi_aoi, '2017-11-24', '2017-12-24', 'mwi_floods_03-18')

# Get ndvi
mwi_ndvi = begin_ndvi_export(mwi_aoi, '2017-11-24', '2017-12-24', 'mwi_floods_03-18')

# Get evap
mwi_evap = begin_evap_export(mwi_aoi, '2017-11-24', '2017-12-24', 'mwi_floods_03-18')

# Get clim
mwi_clip = begin_clim_export(mwi_aoi, '2017-11-24', '2017-12-24', 'mwi_floods_03-18')


################   Mozambique 03-19   #########################################

    
## Landdcover 

modis = ee.ImageCollection('MODIS/006/MCD12Q1')
# choose dates
modis = modis.filterDate('2018-01-01','2019-01-01')

# setting the Area of Interest (AOI)
moz_aoi = ee.Geometry.Rectangle([ 33.69213642, -20.57556134,  35.30819006, -18.97632074])

modis_aoi = modis.filterBounds(moz_aoi)


# the least cloudy image
least_cloudy = ee.Image(modis.sort('CLOUD_COVER').first())
# how cloudy is it?
print('Cloud Cover (%):', least_cloudy.get('CLOUD_COVER').getInfo())


# when was this image taken?
date = ee.Date(least_cloudy.get('system:time_start'))
time = date.getInfo()['value']/1000.
dt.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')

parameters = {'min': 0,
              'max': 1000,
              'dimensions': 512,
              'bands': 'LC_Type1',
              'region': moz_aoi}
             

task_config = {
    'scale': 30,  
    'region': moz_aoi['coordinates'][0]
    }

task = ee.batch.Export.image(least_cloudy, description='exportExample', **task_config)

task.start()




task = ee.batch.Export.image.toDrive(image=least_cloudy,
                                      region=moz_aoi.getInfo()['coordinates'],
                                      description='my_description',
                                      folder='mozam_03/19',
                                      fileNamePrefix='moz_lc_type1',
                                      scale=30,
                                      crs='EPSG:4326')
task.start()


    
#### Precip

trmm = ee.ImageCollection("TRMM/3B42")
trmm = trmm.select(['precipitation', 'HQprecipitation', 'IRprecipitation'])


trmm_1 = trmm.filterDate('2019-04-01','2019-04-09')

trmm_2 = trmm.filterDate('2019-03-24','2019-04-01')

trmm_3 = trmm.filterDate('2019-03-17','2019-03-24')

trmm_4 = trmm.filterDate('2019-03-10','2019-03-17')

trmm_5 = trmm.filterDate('2019-03-03','2019-03-10')

trmm_6 = trmm.filterDate('2019-02-24','2019-03-03')

trmm_7 = trmm.filterDate('2019-02-17','2019-02-24')

trmm_8 = trmm.filterDate('2019-02-10','2019-02-17')

trmm_9 = trmm.filterDate('2019-02-03','2019-02-10')

trmm_10 = trmm.filterDate('2019-01-27','2019-02-03')

trmm_11 = trmm.filterDate('2019-01-20','2019-01-27')

trmm_12 = trmm.filterDate('2019-01-13','2019-01-20')

trmm_13 = trmm.filterDate('2019-01-06','2019-01-13')

trmm_14 = trmm.filterDate('2018-12-30','2019-01-06')

trmm_15 = trmm.filterDate('2018-12-23','2018-12-30')

trmm_16 = trmm.filterDate('2018-12-16','2018-12-23')

trmm_17 = trmm.filterDate('2018-12-09','2018-12-16')



# setting the Area of Interest (AOI)
moz_aoi = ee.Geometry.Rectangle([ 33.69213642, -20.57556134,  35.30819006, -18.97632074])

trmm_list = [trmm_1, trmm_2, trmm_3, trmm_4, trmm_5, trmm_6, trmm_7, trmm_8, trmm_9,
              trmm_10, trmm_11, trmm_12, trmm_13, trmm_14, trmm_15, trmm_16, trmm_17]

# Get precipitations
moz_precip_mean = begin_trmm_export(moz_aoi, trmm_list, 'mean', 'moz_floods_03-19')
moz_precip_total = begin_trmm_export(moz_aoi, trmm_list, 'total', 'moz_floods_03-19')
moz_precip_var = begin_trmm_export(moz_aoi, trmm_list, 'var', 'moz_floods_03-19')


# Get landcover
moz_lc = begin_lc_export(moz_aoi, '2018-01-01', '2019-01-01', 'moz_floods_03-19')

# Get smap
moz_smap = begin_smap_export(moz_aoi, '2018-12-27', '2019-01-27', 'moz_floods_03-19')

# Get ndvi
moz_ndvi = begin_ndvi_export(moz_aoi, '2018-12-27', '2019-01-27', 'moz_floods_03-19')

# Get evap
moz_evap = begin_evap_export(moz_aoi, '2018-12-27', '2019-01-27', 'moz_floods_03-19')

# Get clim
moz_clip = begin_clim_export(moz_aoi, '2018-12-27', '2019-01-27', 'moz_floods_03-19')


################   Mozambique 01-19   #########################################


    
### Landdcover 

modis = ee.ImageCollection('MODIS/006/MCD12Q1')
# choose dates

trmm = ee.ImageCollection("TRMM/3B42")
trmm = trmm.select(['precipitation', 'HQprecipitation', 'IRprecipitation'])

trmm_1 = trmm.filterDate('2019-02-07','2019-02-14')

trmm_2 = trmm.filterDate('2019-01-31','2019-02-07')

trmm_3 = trmm.filterDate('2019-01-24','2019-01-31')

trmm_4 = trmm.filterDate('2019-01-17','2019-01-24')

trmm_5 = trmm.filterDate('2019-01-10','2019-01-17')

trmm_6 = trmm.filterDate('2019-01-03','2019-01-10')

trmm_7 = trmm.filterDate('2018-12-27','2019-01-03')

trmm_8 = trmm.filterDate('2018-12-20','2018-12-27')

trmm_9 = trmm.filterDate('2018-12-13','2018-12-20')

trmm_10 = trmm.filterDate('2018-12-06','2018-12-13')

trmm_11 = trmm.filterDate('2018-11-30','2018-12-06')

trmm_12 = trmm.filterDate('2018-11-23','2018-11-30')

trmm_13 = trmm.filterDate('2018-11-16','2018-11-23')

trmm_14 = trmm.filterDate('2018-11-09','2018-11-16')

trmm_15 = trmm.filterDate('2018-11-02','2018-11-09')

trmm_16 = trmm.filterDate('2018-10-25','2018-11-02')

trmm_17 = trmm.filterDate('2018-10-18','2018-10-25')



# setting the Area of Interest (AOI)
moz_aoi = ee.Geometry.Rectangle([ 32.855, -19.485,  37.115, -15.625])

trmm_list = [trmm_1, trmm_2, trmm_3, trmm_4, trmm_5, trmm_6, trmm_7, trmm_8, trmm_9,
              trmm_10, trmm_11, trmm_12, trmm_13, trmm_14, trmm_15, trmm_16, trmm_17]

# Get precipitations
moz_precip_mean = begin_trmm_export(moz_aoi, trmm_list, 'mean', 'moz_floods_01-19')
moz_precip_total = begin_trmm_export(moz_aoi, trmm_list, 'total', 'moz_floods_01-19')
moz_precip_var = begin_trmm_export(moz_aoi, trmm_list, 'var', 'moz_floods_01-19')


# Get landcover
moz_lc = begin_lc_export(moz_aoi, '2018-01-01', '2019-01-01', 'moz_floods_01-19')

# Get smap
moz_smap = begin_smap_export(moz_aoi, '2018-11-06', '2018-12-06', 'moz_floods_01-19')



smap = ee.ImageCollection("NASA_USDA/HSL/SMAP_soil_moisture")
# choose dates
smap = smap.filterDate('2018-11-06', '2018-12-06')
# Choose band
smap = smap.select(['ssm', 'susm'])

aoi_bound = smap.filterBounds(moz_aoi)

# the least cloudy image
least_cloudy = ee.Image(aoi_bound.sort('CLOUD_COVER').first())
# how cloudy is it?
print('Cloud Cover (%):', least_cloudy.get('CLOUD_COVER').getInfo())        
# when was this image taken?
date = ee.Date(least_cloudy.get('system:time_start'))
time = date.getInfo()['value']/1000.
dt.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
task = ee.batch.Export.image.toDrive(image=least_cloudy,
                                      region=moz_aoi.getInfo()['coordinates'],
                                      description='moz_floods_01' + '_smap',
                                      folder='moz_floods_01',
                                      fileNamePrefix='smap',
                                      maxPixels=1e5,
                                      scale=30,
                                      crs='EPSG:4326')
task.start()



# Get NDVI
moz_ndvi = begin_ndvi_export(moz_aoi, '2018-11-06', '2018-12-06', 'moz_floods_01-19')






ndvi = ee.ImageCollection('MODIS/006/MOD13A2')
# choose dates
ndvi = ndvi.filterDate('2018-11-06', '2018-12-06')
ndvi = ndvi.select(['NDVI', 'EVI'])

# setting the Area of Interest (AOI)
#moz_aoi = ee.Geometry.Rectangle(aoi)
aoi_bound = ndvi.filterBounds(moz_aoi)

# the least cloudy image
least_cloudy = ee.Image(aoi_bound.sort('CLOUD_COVER').first())
# how cloudy is it?
print('Cloud Cover (%):', least_cloudy.get('CLOUD_COVER').getInfo())        
# when was this image taken?
date = ee.Date(least_cloudy.get('system:time_start'))
time = date.getInfo()['value']/1000.
dt.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
task = ee.batch.Export.image.toDrive(image=least_cloudy,
                                      region=moz_aoi.getInfo()['coordinates'],
                                      description='moz_floods_01-19' + '_ndvi',
                                      folder='moz_floods_01-19',
                                      fileNamePrefix='ndvi',
                                      maxPixels=1e9,
                                      scale=30,
                                      crs='EPSG:4326')
task.start()




# Get evap
moz_evap = begin_evap_export(moz_aoi, '2018-11-06', '2018-12-06', 'moz_floods_01-19')


evap = ee.ImageCollection("MODIS/006/MOD16A2")

# choose dates
evap = evap.filterDate('2018-11-06', '2018-12-06')
evap = evap.select('ET')

# setting the Area of Interest (AOI)
aoi_bound = evap.filterBounds(moz_aoi)

# the least cloudy image
least_cloudy = ee.Image(aoi_bound.sort('CLOUD_COVER').first())
# how cloudy is it?
print('Cloud Cover (%):', least_cloudy.get('CLOUD_COVER').getInfo())        
# when was this image taken?
date = ee.Date(least_cloudy.get('system:time_start'))
time = date.getInfo()['value']/1000.
dt.utcfromtimestamp(time).strftime('%Y-%m-%d %H:%M:%S')
task = ee.batch.Export.image.toDrive(image=least_cloudy,
                                      region=moz_aoi.getInfo()['coordinates'],
                                      description='moz_floods_01-19' + '_evap',
                                      folder='moz_floods_01-19',
                                      fileNamePrefix='evap',
                                      maxPixels=1e9,
                                      scale=30,
                                      crs='EPSG:4326')
task.start()



# Get clim
moz_clip = begin_clim_export(moz_aoi, '2018-11-06', '2018-12-06', 'moz_floods_01-19')


######################     Malawi 01-19    ####################################



### Landdcover 

modis = ee.ImageCollection('MODIS/006/MCD12Q1')

# setting the Area of Interest (AOI)
mwi_aoi = ee.Geometry.Rectangle([ 34.325, -17.035,  35.805, -15.335])


trmm = ee.ImageCollection("TRMM/3B42")
trmm = trmm.select(['precipitation', 'HQprecipitation', 'IRprecipitation'])

trmm_1 = trmm.filterDate('2019-02-07','2019-02-14')

trmm_2 = trmm.filterDate('2019-01-31','2019-02-07')

trmm_3 = trmm.filterDate('2019-01-24','2019-01-31')

trmm_4 = trmm.filterDate('2019-01-17','2019-01-24')

trmm_5 = trmm.filterDate('2019-01-10','2019-01-17')

trmm_6 = trmm.filterDate('2019-01-03','2019-01-10')

trmm_7 = trmm.filterDate('2018-12-27','2019-01-03')

trmm_8 = trmm.filterDate('2018-12-20','2018-12-27')

trmm_9 = trmm.filterDate('2018-12-13','2018-12-20')

trmm_10 = trmm.filterDate('2018-12-06','2018-12-13')

trmm_11 = trmm.filterDate('2018-11-30','2018-12-06')

trmm_12 = trmm.filterDate('2018-11-23','2018-11-30')

trmm_13 = trmm.filterDate('2018-11-16','2018-11-23')

trmm_14 = trmm.filterDate('2018-11-09','2018-11-16')

trmm_15 = trmm.filterDate('2018-11-02','2018-11-09')

trmm_16 = trmm.filterDate('2018-10-25','2018-11-02')

trmm_17 = trmm.filterDate('2018-10-18','2018-10-25')


trmm_list = [trmm_1, trmm_2, trmm_3, trmm_4, trmm_5, trmm_6, trmm_7, trmm_8, trmm_9,
              trmm_10, trmm_11, trmm_12, trmm_13, trmm_14, trmm_15, trmm_16, trmm_17]

# Get precipitations
mwi_precip_mean = begin_trmm_export(mwi_aoi, trmm_list, 'mean', 'mwi_floods_01-19')
mwi_precip_total = begin_trmm_export(mwi_aoi, trmm_list, 'total', 'mwi_floods_01-19')
mwi_precip_var = begin_trmm_export(mwi_aoi, trmm_list, 'var', 'mwi_floods_01-19')


# Get landcover
mwi_lc = begin_lc_export(mwi_aoi, '2018-01-01', '2019-01-01', 'mwi_floods_01-19')

# Get smap
mwi_smap = begin_smap_export(mwi_aoi, '2018-11-06', '2018-12-06', 'mwi_floods_01-19')

mwi_ndvi = begin_ndvi_export(mwi_aoi, '2018-11-06', '2018-12-06', 'mwi_floods_01-19')


# Get evap
mwi_evap = begin_evap_export(mwi_aoi, '2018-11-06', '2018-12-06', 'mwi_floods_01-19')

# Get clim
mwi_clim = begin_clim_export(mwi_aoi, '2018-11-06', '2018-12-06', 'mwi_floods_01-19')


#################  Malawi 03-19 (test)  #######################################


trmm = ee.ImageCollection("TRMM/3B42")
trmm = trmm.select(['precipitation', 'HQprecipitation', 'IRprecipitation'])


trmm_1 = trmm.filterDate('2019-04-01','2019-04-09')

trmm_2 = trmm.filterDate('2019-03-24','2019-04-01')

trmm_3 = trmm.filterDate('2019-03-17','2019-03-24')

trmm_4 = trmm.filterDate('2019-03-10','2019-03-17')

trmm_5 = trmm.filterDate('2019-03-03','2019-03-10')

trmm_6 = trmm.filterDate('2019-02-24','2019-03-03')

trmm_7 = trmm.filterDate('2019-02-17','2019-02-24')

trmm_8 = trmm.filterDate('2019-02-10','2019-02-17')

trmm_9 = trmm.filterDate('2019-02-03','2019-02-10')

trmm_10 = trmm.filterDate('2019-01-27','2019-02-03')

trmm_11 = trmm.filterDate('2019-01-20','2019-01-27')

trmm_12 = trmm.filterDate('2019-01-13','2019-01-20')

trmm_13 = trmm.filterDate('2019-01-06','2019-01-13')

trmm_14 = trmm.filterDate('2018-12-30','2019-01-06')

trmm_15 = trmm.filterDate('2018-12-23','2018-12-30')

trmm_16 = trmm.filterDate('2018-12-16','2018-12-23')

trmm_17 = trmm.filterDate('2018-12-09','2018-12-16')



# setting the Area of Interest (AOI)
mwi_aoi = ee.Geometry.Rectangle([ 34.325, -17.035,  35.805, -15.335])

trmm_list = [trmm_1, trmm_2, trmm_3, trmm_4, trmm_5, trmm_6, trmm_7, trmm_8, trmm_9,
              trmm_10, trmm_11, trmm_12, trmm_13, trmm_14, trmm_15, trmm_16, trmm_17]

# Get precipitations
mwi_precip_mean = begin_trmm_export(mwi_aoi, trmm_list, 'mean', 'mwifloods_03-19')
mwi_precip_total = begin_trmm_export(mwi_aoi, trmm_list, 'total', 'mwi_floods_03-19')
mwi_precip_var = begin_trmm_export(mwi_aoi, trmm_list, 'var', 'mwi_floods_03-19')


# Get landcover
mwi_lc = begin_lc_export(mwi_aoi, '2018-01-01', '2019-01-01', 'mwi_floods_03-19')

# Get smap
mwi_smap = begin_smap_export(mwi_aoi, '2018-12-27', '2019-01-27', 'mwi_floods_03-19')

mwi_ndvi = begin_ndvi_export(mwi_aoi, '2018-12-27', '2019-01-27', 'mwi_floods_03-19')

mwi_evap = begin_evap_export(mwi_aoi, '2018-12-27', '2019-01-27', 'mwi_floods_03-19')

# Get clim
mwi_clim = begin_clim_export(mwi_aoi, '2018-12-27', '2019-01-27', 'mwi_floods_03-19')




    
    
