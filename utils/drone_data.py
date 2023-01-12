import numpy as np
#from sqlalchemy import over
import xarray
import rasterio
import rasterio.mask
from rasterio.windows import from_bounds
import rioxarray as rio
import os
import glob

from . import data_processing
from .plt_functions import plot_multibands_fromxarray


import matplotlib.pyplot as plt

from . import classification_functions as clf
import pandas as pd
import geopandas as gpd

from . import gis_functions as gf
from .xr_functions import split_xarray_data

import re

VEGETATION_INDEX = {# rgb bands
'grvi': '(green - red)/(green + red)',
'grvi_eq': '(green_eq - red_eq)/(green_eq + red_eq)',
'mgrvi': '((green*green) - (red*red))/((green*green) + (red*red))',
'rgbvi': '((green*green) - (blue*red))/ ((green*green) + (blue*red))',
 # nir indexes
'ndvi': '(nir - red)/(nir + red)',
'ndre': '(nir - edge)/(nir + edge)',
'gndvi': '(nir - green)/(nir + green)',
'regnvi': '(edge - green)/(edge + green)',
'reci': '(nir / edge) - 1',
'negvi': '((nir*nir) - (edge*green))/ ((nir*nir) + (edge*green))'}



def drop_bands(xarraydata, bands):
    for i in bands:
        xarraydata = xarraydata.drop(i)
    
    return xarraydata

def _solve_red_edge_order(listpaths, bands):

    ordered =[]
    for band in bands:
        for src in listpaths:
            if band in src and band not in ordered:
                if "red" in src and "red" == band:
                    if "edge" not in src:
                        ordered.append(src)
                else:
                    ordered.append(src)

    return ordered

def filter_list(list1, list2):
    list_filtered = []
    for strlist2 in list2:
        for strlist1 in list1:
            if strlist2 in strlist1:
                if strlist1 not in list_filtered:
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
        if "edge" in bands:
            imgfiles_filtered = _solve_red_edge_order(imgfiles_filtered, bands)

        return imgfiles_filtered

    except ValueError:
        print("file path doesn't exist")

def calculate_vi_fromxarray(xarraydata, vi='ndvi', expression=None, label=None):

    variable_names = list(xarraydata.keys())
    if expression is None and vi in list(VEGETATION_INDEX.keys()):
        expression = VEGETATION_INDEX[vi]

    namask = xarraydata.attrs['nodata']
    # modify expresion finding varnames
    symbolstoremove = ['*','-','+','/',')','.','(',' ','[',']']
    test = expression
    for c in symbolstoremove:
        test = test.replace(c, '-')

    test = re.sub('\d', '-', test)
    varnames = [i for i in np.unique(np.array(test.split('-'))) if i != '']
    for i, varname in enumerate(varnames):
        
        if varname in variable_names:
            exp = (['listvar[{}]'.format(i), varname])
            expression = expression.replace(exp[1], exp[0])
        else:
            raise ValueError('there is not a variable named as {}'.format(varname))

    listvar = []
    if vi not in variable_names:

        for i, varname in enumerate(varnames):
            if varname in variable_names:
                varvalue = xarraydata[varname].data
                varvalue[varvalue == namask] = np.nan
                listvar.append(varvalue)
        
        vidata = eval(expression)
            
        if label is None:
            label = vi

        vidata[np.isnan(vidata)] = xarraydata.attrs['nodata']
        vidata[vidata == namask] = np.nan
        xrvidata = xarray.DataArray(vidata)
        
        xrvidata.name = label
        
        dimsnames = list(xarraydata.dims.keys())
        if xrvidata.ndim == 3 and len(dimsnames) > 2:
        
            dim1name = [i for i in dimsnames if len(xarraydata[i].values) == xrvidata.shape[0]][0]
            dim2name = [i for i in dimsnames if len(xarraydata[i].values) == xrvidata.shape[1]][0]
            dim3name = [i for i in dimsnames if i not in (dim1name,dim2name)]
            xrvidata = xrvidata.rename(dict(zip(xrvidata.dims,
                                                [dim1name,dim2name] + dim3name)))
        
        else:
            xrvidata = xrvidata.rename(dict(zip(xrvidata.dims,
                                            list(xarraydata.dims.keys()))))


        #print(xrvidata.dims.keys())
        xarraydata = xarray.merge([xarraydata, xrvidata])
        #xarraydata = xrvidata

        xarraydata.attrs['count'] = len(list(xarraydata.keys()))

    else:
        print("the VI {} was calculated before {}".format(vi, variable_names))

    return xarraydata


def multiband_totiff(xrdata, filename, varnames=None):

    #varnames = self._checkbandstoexport(varnames)
    metadata = xrdata.attrs

    if filename.endswith('tif'):
        suffix = filename.index('tif')
    else:
        suffix = (len(filename) + 1)

    if len(varnames) > 1:
        metadata['count'] = len(varnames)
        fn = "{}{}.tif".format(filename[:(suffix - 1)], "_".join(varnames))
        xrdata.rio.to_raster(fn)

class UAVPlots():
    pass

class DroneData:
    """
    Handles UAV images using xarray package.
    Attributes
    ----------
    drone_data : xarray
        UAV image in xarray format.
    variable_names : list str
        Spectral bands that compose the uav iamge.
    

    """

    @property
    def available_vi():
        return VEGETATION_INDEX


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
        """
        function to calculate vegetation indices

        Parameters:
        ----------
        vi : str
            vegetation index name, if the vegetatio index is into the list, it will compute it using the equation, otherwise it will be necessary to prodive it
        expression: str, optional
            equation to calculate the vegetation index, eg (nir - red)/(nir + red)
        
        Return:
        ----------
        None
        """

        if vi == 'ndvi':
            if 'nir' in self.variable_names:
                expression = '(nir - red) / (nir + red)' 
            else:
                raise ValueError('It was not possible to calculate ndvi as default, please provide equation')

        elif expression is None:
            raise ValueError('please provide a equation to calculate this index: {}'.format(vi))

        self.drone_data = calculate_vi_fromxarray(self.drone_data, vi, expression, label)

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
                            crs=None,
                            bands=None, 
                            long_direction=True):
        """
        function to extract data using coordinates

        Parameters:
        ----------
        points : list
            a list of coordinates with values in latitude and longitude
        bands: list, optional
            a list iwht the na,es of the spectral bands for extracting the data
        crs: str, optional
            
        
        Return:
        ----------
        None
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

    def tif_toxarray(self, multiband=False, bounds = None):

        riolist = []
        imgindex = 1
        nodata = None
        boundswindow = None
        
        for band, path in zip(self._bands, self._files_path):
            
            with rasterio.open(path) as src:

                tr = src.transform
                nodata = src.nodata
                metadata = src.profile.copy()
                if bounds is not None:
                    #boundswindow = from_bounds(bounds[0],bounds[1],bounds[2],bounds[3], src.transform)
                    #tr = src.window_transform(boundswindow)
                    img, tr = rasterio.mask.mask(src, bounds, crop=True)
                   
                    img = img[(imgindex-1),:,:]
                    img = img.astype(float)
                    img[img == nodata] = np.nan
                    nodata = np.nan

                else:
                    img = src.read(imgindex, window = boundswindow)
                    
                
                metadata.update({
                    'height': img.shape[0],
                    'width': img.shape[1],
                    'transform': tr})

            if img.dtype == 'uint8':
                img = img.astype(float)
                metadata['dtype'] == 'float'


            xrimg = xarray.DataArray(img)
            xrimg.name = band
            riolist.append(xrimg)

            if multiband:
                imgindex += 1

        # update nodata attribute
        metadata['nodata'] = nodata

        multi_xarray = xarray.merge(riolist)
        multi_xarray.attrs = metadata

        ## assign coordinates
        #tmpxr = xarray.open_rasterio(self._files_path[0])
        xvalues, yvalues = gf.xy_fromtransform(metadata['transform'], metadata['width'],metadata['height'])

        multi_xarray = multi_xarray.assign_coords(x=xvalues)
        multi_xarray = multi_xarray.assign_coords(y=yvalues)
        
        multi_xarray = multi_xarray.rename({'dim_0': 'y', 'dim_1': 'x'})
        metadata['count'] = len(multi_xarray.keys())

        return multi_xarray

    def plot_multiplebands(self, bands, height=20, width=14, xinverse = False):
        return plot_multibands_fromxarray(self.drone_data, bands, height, width, xinverse = xinverse)

    def plot_singleband(self, band, height=12, width=8):

        # Define a normalization from values -> colors

        datatoplot = self.drone_data[band].data
        datatoplot[datatoplot == self.drone_data.attrs['nodata']] = np.nan
        fig, ax = plt.subplots(figsize=(height, width))

        im = ax.imshow(datatoplot)
        fig.colorbar(im, ax=ax)
        ax.set_axis_off()
        plt.show()


    def to_tiff(self, filename, channels='all',multistack = False):
        """
        Using this function the drone data will be saved as a tiff element in a given 
        filepath.
        ```
        Args:
            filename: The file name path that will be used to save the spatial data.
            channels: optional, the user can select the exporting channels.
            multistack: boolean, default False, it will export the inofrmation as a multistack array or layer by layer

        Returns:
            NONE
        """
        varnames = self._checkbandstoexport(channels)
        metadata = self.drone_data.attrs

        if filename.endswith('tif'):
            suffix = filename.index('tif')
        else:
            suffix = (len(filename) + 1)
        if multistack:
            multiband_totiff(self.drone_data, filename, varnames= varnames)
        else:
            if len(varnames) > 0:
                for i, varname in enumerate(varnames):
                    imgtoexport = self.drone_data[varname].data.copy()
                    fn = "{}_{}.tif".format(filename[:(suffix - 1)], varname)
                    with rasterio.open(fn, 'w', **metadata) as dst:
                        dst.write_band(1, imgtoexport)

            else:
                print('check the bands names that you want to export')

    def split_into_tiles(self, polygons=False, **kargs):

        self._tiles_pols = split_xarray_data(self.drone_data, polygons=polygons, **kargs)
        
        print("the image was divided into {} tiles".format(len(self._tiles_pols)))
        
    def clip_using_gpd(self, gpd_df, replace = True):
        clipped = gf.clip_xarraydata(self.drone_data, gpd_df)
        if replace:
            self.drone_data = clipped
        else:
            return clipped

    def tiles_data(self,id_tile):

        if self._tiles_pols is not None:
            window, transform = self._tiles_pols[id_tile]
            xrmasked = gf.crop_using_windowslice(self.drone_data, window, transform)

        else:
            raise ValueError("Use split_into_tiles first")

        return xrmasked

    def __init__(self,
                 inputpath,
                 bands=None,
                 multiband_image=False,
                 roi = None,
                 table=False,
                 bounds = None):

        if bands is None:
            self._bands = ['red', 'green', 'blue']
        else:
            self._bands = bands

        
        self._clusters = np.nan
        self._tiles_pols = None

        if not multiband_image:
            self._files_path = get_files_paths(inputpath, self._bands)

        else:
            if inputpath.endswith('.tif'):
                self._files_path = [inputpath for i in range(len(self._bands))]
            else:
                try:
                    imgfiles = glob.glob(inputpath + "*.tif")[0]
                except:
                    raise ValueError(f"There are no tiff files in this dir {inputpath}")
                    
                self._files_path = [imgfiles for i in range(len(self._bands))]


        if len(self._files_path)>0:
            self.drone_data = self.tif_toxarray(multiband_image, bounds=bounds)
        else:
            raise ValueError('Non file path was found')
            
        if roi is not None:
            self.clip_using_gpd(roi, replace = True)

        if table:
            self._data, self._nanindex = self.data_astable()



