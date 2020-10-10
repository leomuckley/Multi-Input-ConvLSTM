#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 10:11:42 2020

@author: leo
"""


import os
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling


class CRS():
    """ Convert TIFF image coordinate system.
    
    For example, if the existing TIFF image has crs = EPSG:2935 it is possible
    to convert this to EPSG:4326 and output the new image.
    -------------------------
    # File to be converted
    crs = CRS("data/out.tif")
    crs.convert()
    
    # Read converted (4326) file
    new_file = rio.open("data/out-4326.tif")
    print(new_file.meta) # 4326
    -------------------------

    """
    
    def __init__(self, filename):
        self.crs = "EPSG:4326"
        #self.path = path
        self.old_filename = filename
        #root_ext = os.path.splitext(filename)
        #self.new_filename = root_ext[0] + f"-{self.crs[-4:]}" + root_ext[1]
        
    def convert(self, new_filename):
        with rio.open(self.old_filename) as src:
            transform, width, height = calculate_default_transform(
                src.crs, self.crs, src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': self.crs,
                'transform': transform,
                'width': width,
                'height': height
            })
            
            with rio.open(new_filename, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rio.band(src, i),
                        destination=rio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=self.crs,
                        resampling=Resampling.nearest)
        print(f"Converted tiff file to {self.crs}")
        return None
    
   