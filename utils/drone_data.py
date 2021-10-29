import numpy as np
import xarray
import rasterio
import os
import glob

from utils import data_processing
from utils.plt_functions import plot_multibands_fromxarray

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
from utils import classification_functions as clf
import pandas as pd
import geopandas as gpd

from utils import gis_functions as gf

def filter_list(list1, list2):
    list_filtered = []
    for strlist2 in list2:
        for strlist1 in list1:
            if strlist2 in strlist1:
                list_filtered.append(strlist1)

    return list_filtered


def normalized_difference(array1, array2, namask=np.nan):
    if np.logical_not(np.isnan(namask)):
        array1[array1 == namask] = np.nan
        array2[array2 == namask] = np.nan

    return ((array1 - array2) /
            (array1 + array2))


def get_files_paths(path, bands):
    try:

        imgfiles = glob.glob(path + "*.tif")
        imgfiles_filtered = filter_list(imgfiles, bands)

        return imgfiles_filtered

    except ValueError:
        print("file path doesn't exist")


class DroneData:

    @property
    def variable_names(self):
        return list(self.drone_data.keys())


    def _checkbandstoexport(self, bands):

        if bands == 'all':
            bands = self.variable_names

        elif not isinstance(bands, list):
            bands = [bands]

        bands = [i for i in bands if i in self.variable_names]

        return bands


    def add_layer(self, fn, variable_name):
        with rasterio.open(fn) as src:
            xrimg = xarray.DataArray(src.read(1))

        xrimg.name = variable_name
        xrimg = xrimg.rename({'dim_0': 'y', 'dim_1': 'x'})

        self.drone_data = xarray.merge([self.drone_data, xrimg])

    def data_astable(self):

        npdata2dclean, idsnan = data_processing.from_xarray_to_table(self.drone_data,
                                                                     nodataval=self.drone_data.attrs['nodata'])

        return [npdata2dclean, idsnan]

    def calculate_vi(self, vi='ndvi', expression=None, label=None):
        namask = self.drone_data.attrs['nodata']
        if 'blue' in self.variable_names:
            blue = self.drone_data.blue.data
            blue[blue == namask] = np.nan
        if 'green' in self.variable_names:
            green = self.drone_data.green.data
            green[green == namask] = np.nan
        if 'red' in self.variable_names:
            red = self.drone_data.red.data
            red[red == namask] = np.nan
        if 'nir' in self.variable_names:
            nir = self.drone_data.nir.data
            nir[nir == namask] = np.nan
        if 'r_edge' in self.variable_names:
            r_edge = self.drone_data.r_edge.data
            r_edge[r_edge == namask] = np.nan

        if expression is not None:
            vidata = eval(expression)
            if label is None:
                label = vi

        elif vi == 'ndvi':
            vidata = normalized_difference(nir, red, self.drone_data.attrs['nodata'])
            label = 'ndvi'

        elif vi == 'ndvire':
            vidata = normalized_difference(nir, r_edge, self.drone_data.attrs['nodata'])
            label = 'ndvire'

        vidata[np.isnan(vidata)] = self.drone_data.attrs['nodata']

        xrvidata = xarray.DataArray(vidata)
        xrvidata.name = label
        xrvidata = xrvidata.rename({'dim_0': 'y', 'dim_1': 'x'})

        self.drone_data = xarray.merge([self.drone_data, xrvidata])

    def rf_classification(self, model, features=None):

        if features is None:
            features = ['blue', 'green', 'red',
                        'r_edge', 'nir', 'ndvi', 'ndvire']

        img_clas = clf.img_rf_classification(self.drone_data, model, features)
        img_clas = xarray.DataArray(img_clas)
        img_clas.name = 'rf_classification'

        self.drone_data = xarray.merge([self.drone_data, img_clas])


    def clusters(self, nclusters=2, method="kmeans", p_sample=10, pcavariance=0.5):
        # preprocess data
        data = self._data
        idsnan = self._nanindex

        if method == "kmeans":
            nsample = int(np.round(data.shape[0] * (p_sample / 100)))
            clusters = clf.kmeans_images(data,
                                         nclusters,
                                         nrndsample=nsample, eigmin=pcavariance)

        climg = data_processing.assign_valuestoimg((clusters['labels'] + 1),
                                                   self.drone_data.dims['y'],
                                                   self.drone_data.dims['x'], idsnan)


        climg = xarray.DataArray(climg)
        climg.name = 'clusters'

        self.drone_data = xarray.merge([self.drone_data, climg])
        self._clusters = clusters

    def extract_usingpoints(self, points,
                            bands=None, crs=None,
                            long_direction=True):
        """

        :param points:
        :param bands:
        :param crs:
        :param long_direction:
        :return:
        """

        if bands is None:
            bands = self.variable_names
        if crs is None:
            crs = self.drone_data.attrs['crs']

        if type(points) == str:
            coords = pd.read_csv(points)

        elif type(points) == list:
            if np.array(points).ndim == 1:
                points = [points]

            coords = pd.DataFrame(points)

        geopoints = gpd.GeoDataFrame(coords,
                                     geometry=gpd.points_from_xy(coords.iloc[:, 0],
                                                                 coords.iloc[:, 1]),
                                     crs=crs)

        return gf.get_data_perpoints(self.drone_data.copy(),
                                     geopoints,
                                     bands,
                                     long=long_direction)

    def tif_toxarray(self, multiband = False):

        riolist = []
        imgindex = 1
        for band, path in zip(self._bands, self._files_path):
            

            with rasterio.open(path) as src:
                img = src.read(imgindex)
                nodata = src.nodata
                metadata = src.profile.copy()

            if img.dtype == 'uint8':
                img = img.astype(float)
                metadata['dtype'] == 'float'
                nodata= None

            xrimg = xarray.DataArray(img)
            xrimg.name = band
            riolist.append(xrimg)

            if multiband:
                imgindex += 1

        

        # update nodata attribute
        metadata['nodata'] = nodata
        metadata['count'] = self._bands

        multi_xarray = xarray.merge(riolist)
        multi_xarray.attrs = metadata

        ## assign coordinates
        tmpxr = xarray.open_rasterio(self._files_path[0])

        multi_xarray = multi_xarray.assign_coords(x=tmpxr['x'].data)
        multi_xarray = multi_xarray.assign_coords(y=tmpxr['y'].data)
        
        multi_xarray = multi_xarray.rename({'dim_0': 'y', 'dim_1': 'x'})

        return multi_xarray

    def plot_multiplebands(self, bands, fig_sizex=12, fig_sizey=8):
        return plot_multibands_fromxarray(self.drone_data, bands, fig_sizex, fig_sizey)


    def plot_singleband(self, band, height=12, width=8):

        # Define a normalization from values -> colors

        datatoplot = self.drone_data[band].data
        datatoplot[datatoplot == self.drone_data.attrs['nodata']] = np.nan
        fig, ax = plt.subplots(figsize=(height, width))

        ax.imshow(datatoplot)

        ax.set_axis_off()
        plt.show()


    def multiband_totiff(self, filename, varnames='all'):

        varnames = self._checkbandstoexport(varnames)
        metadata = self.drone_data.attrs

        if filename.endswith('tif'):
            suffix = filename.index('tif')
        else:
            suffix = (len(filename) + 1)

        if len(varnames) > 1:
            metadata['count'] = len(varnames)
            imgstoexport = []
            fname = ""
            for varname in varnames:
                fname = fname + "_" + varname
                imgstoexport.append(self.drone_data[varname].data.copy())

            fn = "{}{}.tif".format(filename[:(suffix - 1)], fname)
            imgstoexport = np.array(imgstoexport)
            with rasterio.open(fn, 'w', **metadata) as dst:
                for id, layer in enumerate(imgstoexport, start=1):
                    dst.write_band(id, layer)

    def to_tiff(self, filename, varnames='all'):

        varnames = self._checkbandstoexport(varnames)
        metadata = self.drone_data.attrs

        if filename.endswith('tif'):
            suffix = filename.index('tif')
        else:
            suffix = (len(filename) + 1)

        if len(varnames) > 0:
            for i, varname in enumerate(varnames):
                imgtoexport = self.drone_data[varname].data.copy()
                fn = "{}_{}.tif".format(filename[:(suffix - 1)], varname)
                with rasterio.open(fn, 'w', **metadata) as dst:
                    dst.write_band(1, imgtoexport)

        else:
            print('check the bands names that you want to export')
    
    def split_into_tiles(self, polygons = False, **kargs):
        tilesdata = gf.split_xarray_data(self.drone_data, polygons = polygons, **kargs)
        print("the image wass diveded into {} tiles".format(len(tilesdata)))
        self.tiles_data = tilesdata

    def __init__(self,
                 inputpath,
                 bands=None,
                 multiband_image = False,
                 table = True):

        if bands is None:
                self._bands = ['red', 'green', 'blue']
        else:
            self._bands = bands

        self._clusters = np.nan
        if not multiband_image:
            self._files_path = get_files_paths(inputpath, self._bands)
            
        else:
            if inputpath.endswith('.tif'):
                self._files_path = [inputpath for i in range(len(self._bands))]

        self.drone_data = self.tif_toxarray(multiband_image)

        if table:
            self._data, self._nanindex = self.data_astable()
