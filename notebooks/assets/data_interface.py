# Copyright 2019 Damian Schori. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import random
import rasterio as rio
import geopandas as gpd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rasterio.mask import mask
from rasterio.plot import show
from shapely.geometry import Polygon
from skimage.transform import rescale, resize
from skimage.color import rgb2gray
from shapely import affinity
from skimage.io import imsave
import skimage
import os

class Dataset():
    def __init__(self, name, 
                 date,
                 rgb_path, 
                 ms_path, 
                 outer_shapefile, 
                 rgb_bands_to_read, 
                 ms_bands_to_read,
                 mask_shapefile = None,
                 grid=None,
                 grid_overlapp=0,
                 slice_shape=(512, 512)):
        self.name = name
        self.date = date
        self.rgb_path = rgb_path
        self.ms_path = ms_path
        self.mask_shapefile = mask_shapefile
        self.outer_shapefile = outer_shapefile
        self.rgb_bands_to_read = rgb_bands_to_read
        self.ms_bands_to_read = ms_bands_to_read
        self.grid = grid
        self.grid_overlapp = grid_overlapp
        self.slice_shape = slice_shape
        self.masked_raster = None
        self.masked_transform = None
        self.initialize()
        
    def initialize(self):
        if self.mask_shapefile is None:
            print('Attention, no maks shapefile defined in Dataset {}'.format(self.name))
        def replace_names(old_name):
            return old_name.replace(old_name[:8], self.name[:8])

        if self.grid is None:
            self.__add_grid()
        else:
            self.grid['name_'] = self.grid.name_.map(replace_names)

        if 'id' in self.grid.columns:
            self.grid = self.grid.drop(columns=['id'])

        self.grid['date'] = self.date

    def visualize(self, with_grid=False, with_mask=False):
        '''
        Shows bands of dataset
        '''
        sns.reset_orig()

        fig, ax = plt.subplots(nrows=1, ncols=len(self.rgb_bands_to_read), figsize=(14, 14))
            
        with rio.open(self.rgb_path) as src:
            
            shapes = [feature.geometry for i, feature in self.outer_shapefile.iterrows()]
            out_image, out_transform = rio.mask.mask(src, shapes, crop=True)

            for i, band in enumerate(self.rgb_bands_to_read):
                show(out_image[i], ax=ax[i], transform=out_transform, cmap='viridis')

                ax[i].set_title('Dataset "{}", Band {}'.format(self.name, band))
                ax[i].axis('off')

                if with_grid:
                    if self.grid is None:
                        raise NotImplementedError('grid not created yet')
                    self.grid.plot(ax=ax[i], edgecolor='black', alpha=0.2)

                if with_mask:
                    if self.mask_shapefile is None:
                        raise NotImplementedError('shapefile not added yet')
                    self.mask_shapefile.plot(ax=ax[i], edgecolor='white', alpha=0.5)

        plt.subplots_adjust(wspace=0.01, hspace=0.1)
        plt.tight_layout()
        plt.show()
        
    def __add_grid(self):
        '''
        Creates Grid with (x, y) sizes in px from outer_shapefile
        outer_shapefile: geopandas shapefile
        length_x: length x in px
        length_y: length y in px
        new_value: new values to overwrite current values
        '''
        with rio.open(self.rgb_path) as src:
            xmin, ymin, xmax, ymax = self.outer_shapefile.total_bounds
            length_y_pixel, length_x_pixel = self.slice_shape
            pixelSizeX = src.transform[0]
            pixelSizeY = -src.transform[4]

            length_x_world = pixelSizeX * length_x_pixel
            length_y_world = pixelSizeY * length_y_pixel

            n_x = ((xmax-xmin) // length_x_world)+1
            n_y = ((ymax-ymin) // length_y_world)+1

            xmax = xmin + n_x*(length_x_world)
            ymax = ymin + n_y*(length_y_world)

            overlapp_x_world = self.grid_overlapp * pixelSizeX
            overlapp_y_world = self.grid_overlapp * pixelSizeY

            sumed_overlapp_x_world = overlapp_x_world * n_x
            sumed_overlapp_y_world = overlapp_y_world * n_y

            rows = list(np.arange(xmin, xmax, (xmax-xmin-sumed_overlapp_x_world)/(n_x)))
            cols = list(np.arange(ymin, ymax, (ymax-ymin-sumed_overlapp_y_world)/(n_y)))

            self.grid = gpd.GeoDataFrame(columns={'geometry', 'grid_id', 'name_'})
            i = 0
            for yi, y in enumerate(cols):
                for xi, x in enumerate(rows):
                    polygon = Polygon([ (x, y),
                                        (x+length_x_world, y),
                                        (x+length_x_world, y+length_y_world),
                                        (x, y+length_y_world)])
                    self.grid = self.grid.append({'geometry': polygon,
                                                  'grid_id': "x{}_y{}".format(xi+1000, yi+1000),
                                                  'outer_bounds': polygon,
                                                  'name_': "{}_{}".format(self.name, i)}, ignore_index=True)
                    i += 1
                    
            self.grid = gpd.overlay(self.outer_shapefile, self.grid, how='intersection')

        
class Data_Interface():
    def __init__(self, datasets, encoding):
        self.datasets = datasets
        self.species_encoding = encoding
        self.df = None
        self.combine_df()

    def info(self):
        print("Datasets in Interface:")
        for dataset in self.datasets:
            print("- {}".format(dataset.name))

    def combine_df(self):
        dfs = []
        for dataset in self.datasets:
            grid_df = dataset.grid.copy()
            dfs.append(grid_df)
        self.df = pd.concat(dfs)

    def stack_mask(self, msk):
        msks = [(msk==channel).astype(float) for channel in range(1, 3)]
        msks = np.stack(msks, axis=-1)
        background = 1 - msks.sum(axis=-1, keepdims=True)
        return np.concatenate((msks, background), axis=-1)

    def save(self, folder_name, skip_black_greater=0.):
        save_path = '../data/sliced/' + folder_name + '/'

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            os.makedirs(save_path+'images')
            os.makedirs(save_path+'masks')

        for i, row in self.df.iterrows():
            img, msk = self.get_pair(grid_id=row.grid_id, date=row.date)
            black_pixels = np.sum(img == 0.0)/(img.shape[0]*img.shape[1]*img.shape[2])
            if black_pixels > skip_black_greater:
                continue
            msk = self.stack_mask(msk)
            imsave(save_path + 'images/' + str(row.date) + "_" + str(row.grid_id) + ".png", skimage.img_as_ubyte(img))
            imsave(save_path + 'masks/' + str(row.date) + "_" + str(row.grid_id) + ".png", skimage.img_as_ubyte(msk))

    def get_pair(self, grid_id='random', date='random', print_info=False):
        '''
        Returns image with all bands added to memory and coressponding mask
        grid_idx: the id of the grid on which to return corresponding image and mask
        returns: image, mask with shape (heigth, width, number_bands), (heigth, width)
        '''
        if date == 'random':
            date = random.sample(self.df.date.unique().tolist(), 1)[0]
        elif date not in self.df.date.unique().tolist():
            raise ValueError(
                "Date {} not found in Datainterface".format(date))

        if grid_id == 'random':
            grid_id = random.sample(self.df.grid_id.unique().tolist(), 1)[0]
        elif grid_id not in self.df.grid_id.tolist():
            raise ValueError(
                "Grid ID {} not found in Datainterface".format(grid_id))
        
        row = self.df.loc[(self.df["grid_id"] == grid_id) & (self.df["date"] == date)]
        
        assert len(row) == 1, "Multiple rows found"
        
        dataset = None
        
        for d in self.datasets:
            if d.date in date:
                dataset = d
                break

        assert dataset is not None, "No dataset found for date {}".format(date)
        
        if dataset.mask_shapefile is not None:
            msk = self.__get_rastered_mask(row, dataset, dataset.mask_shapefile)
        else:
            msk = np.zeros(dataset.slice_shape)

        img_rgb = self.__get_img(row, dataset, 'rgb')
        if dataset.ms_bands_to_read is not None:
            img_ms = self.__get_img(row, dataset, 'ms')
            img = np.concatenate([img_rgb, img_ms], axis=-1)
        else:
            img = img_rgb
        
        if print_info:
            print(date, grid_id)
            
        return img, msk
    
    def get_pair_on_same_date(self, grid_id='random', print_info=False):
        imgs = []
        msks = []

        if grid_id == 'random':
            grid_id = random.sample(self.df.grid_id.unique().tolist(), 1)[0]
        elif grid_id not in self.df.grid_id.unique().tolist():
            raise ValueError(
                "Grid ID {} not found in Dataframe".format(grid_id))
                
        for dataset in self.datasets:
            #row = dataset.grid[dataset.grid['grid_id'].str.match(grid_id)]
            row = dataset.grid.loc[dataset.grid['grid_id'] == grid_id]
            img, msk = self.get_pair(row.grid_id.values[0], row.date.values[0], print_info=print_info)
            imgs.append(img)
            msks.append(msk)

        return np.stack(imgs, axis=0), np.stack(msks, axis=0)
    
    def __get_rastered_mask(self, grid, dataset, shapefile):
        '''
        Raster a single row of daframe with the corresponding bounding box
        row: row of geodataframe
        bbox: bounding box as polygon
        '''
        transform = rio.transform.from_bounds(*grid.outer_bounds.values[0].bounds, *dataset.slice_shape)
        objects =  gpd.overlay(shapefile, grid, how='intersection')
        shapes = ((row.geometry, self.species_encoding[row.Species]) for _, row in objects.iterrows())

        try:
            rastered_shape = rio.features.rasterize(shapes=shapes,
                                                    out_shape=dataset.slice_shape,
                                                    transform=transform)
        except:
            rastered_shape = np.zeros(dataset.slice_shape)

        return rastered_shape

    def __get_img(self, grid, dataset, bands='rgb'):
        '''
        Returns sliced from geo bunds of all bands in memory
        bounds: geo bounds of tuple (left, bottom, right, top)
        returns: imgs, sliced images 
        '''

        shapes = [row.outer_bounds for _, row in grid.iterrows()]
        #return shapes
        if bands == 'rgb':
            with rio.open(dataset.rgb_path) as src:
                out_image, _ = rio.mask.mask(src, shapes, crop=True)
                bands = np.stack([out_image[band] for band in dataset.rgb_bands_to_read], -1)
        elif bands == 'ms':
            with rio.open(dataset.ms_path) as src:
                out_image, _ = rio.mask.mask(src, shapes, crop=True)
                bands = np.stack([out_image[band] for band in dataset.ms_bands_to_read], -1)
        else:
            raise ValueError("Wrong bands keyword, got {}".format(bands))
                
        bands = resize(bands, dataset.slice_shape)
        
        return bands

    def create_prediction(self, model, date):
        for dataset in self.datasets:
            if dataset.date == date:
                prediction = dataset
        for i, row in self.df.iterrows():
            if row.date != date:
                continue
            img, msk = self.get_pair(grid_id=row.grid_id, date=row.date)

