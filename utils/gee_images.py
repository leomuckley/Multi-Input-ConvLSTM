#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
from urllib.parse import urljoin
import pandas as pd

class GeeImages():
    """ Helper function for adding GEE images. """
    
    def __init__(self, filepath, filelist):
        self.filepath = filepath
        self.filelist = [urljoin(self.filepath, i) for i in filelist]      


    def get_stats(self, df, stats_list, return_stats='mean'):
        s = []
        for i in range(len(df)):
            s.append(stats_list[i][return_stats])
                         
        return pd.Series(s)

        
    def add_image(self, df, file):
        image = rasterio.open(file)
        band1 = image.read(1)
        affine = image.transform
        gdf = gpd.GeoDataFrame(df)
        
        stat = zonal_stats(gdf, band1, affine=affine, stats=['mean'])
        file = file.replace('.tif', '')
        file = file.replace(f'{self.filepath}', '')
        df[f'{file}'] = self.get_stats(df, stats_list=stat, return_stats='mean')
        
        # Return new df
        return df
    
        
    def add_all_images(self, df):
        
        for file in self.filelist:
            df = self.add_image(df, file)
            
        return df
        
