import numpy as np
#from sqlalchemy import over
import xarray
import rasterio
import rasterio.mask
from rasterio.windows import from_bounds
import rioxarray as rio
import os
import glob
from PIL import Image

from . import data_processing
from .plt_functions import plot_multibands_fromxarray


import matplotlib.pyplot as plt

from . import classification_functions as clf
import pandas as pd
import geopandas as gpd

from . import gis_functions as gf
from .xr_functions import split_xarray_data, add_2dlayer_toxarrayr
from .general import VEGETATION_INDEX
from .mc_imagery import calculate_vi_fromarray

import re
import pickle

from typing import List, Optional


def drop_bands(xarraydata, bands):
    
    """
    Drops specified bands from an xarray dataset.
    
    Parameters:
    ----------
    xarraydata : xarray.Dataset
        The xarray dataset from which bands will be dropped.
    bands : list
        List of bands to be dropped.
    
    Returns:
    -------
    xarray.Dataset
        The modified xarray dataset after dropping the bands.
    """
    
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
    
    """
    Calculates the normalized difference between two arrays.

    Parameters:
    ----------
    xarraydata : numpy.ndarray
        First input array.
    array2 : numpy.ndarray
        Second input array.
    namask : numeric, optional
        Value to be treated as no-data (NaN) in the calculations.

    Returns:
    -------
    numpy.ndarray
        Array of normalized difference values.
    """
    
    if np.logical_not(np.isnan(namask)):
        array1[array1 == namask] = np.nan
        array2[array2 == namask] = np.nan

    return ((array1 - array2) /
            (array1 + array2))


def get_files_paths(path, bands):
    
    """
    Retrieves file paths for images corresponding to specified bands.

    Parameters:
    ----------
    path : str
        Directory path where image files are located.
    bands : list of str
        List of spectral bands to filter the image files.

    Returns:
    -------
    list of str
        List of filtered image file paths.

    Raises:
    ------
    FileNotFoundError
        If the specified file path does not exist.
    """
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"file path {path} doesn't exist")
    
    
    imgfiles = glob.glob(path + "*.tif")
    
    imgfiles_filtered = filter_list(imgfiles, bands)
    if "edge" in bands:
        imgfiles_filtered = _solve_red_edge_order(imgfiles_filtered, bands)

    return imgfiles_filtered

    


def calculate_vi_fromxarray(xarraydata, vi='ndvi', expression=None, label=None, overwrite = False):
    """
    Calculates vegetation indices from an xarray dataset.

    Parameters:
    ----------
    xarraydata : xarray.Dataset
        The xarray dataset from which to calculate the vegetation index.
    vi : str, default 'ndvi'
        Name of the vegetation index to be calculated.
    expression : str, optional
        Custom expression for calculating the vegetation index.
    label : str, optional
        Label for the new vegetation index data in the dataset.
    overwrite : bool, default False
        If True, overwrite the existing data; otherwise, skip if already present.

    Returns:
    -------
    xarray.Dataset
        The xarray dataset with the new vegetation index data added.
    """
    
    variable_names = list(xarraydata.keys())
    namask = xarraydata.attrs['nodata']
    vidata, label = calculate_vi_fromarray(xarraydata.to_array().values, 
                           variable_names,vi=vi, 
                           expression=expression, 
                           label=label, navalues = namask, overwrite = overwrite)
    
    if vidata is not None:
        vidata[np.isnan(vidata)] = xarraydata.attrs['nodata']
        xarraydata[label] = xarraydata[variable_names[0]].copy()
        
        xarraydata[label].values = vidata
        xarraydata.attrs['count'] = len(list(xarraydata.keys()))

    else:
        print("the VI {} was calculated before {}".format(vi, variable_names))

    return xarraydata


def multiband_totiff(xrdata, filename, varnames=None):

    """
    Converts multiband xarray data to a TIFF file.

    Parameters:
    ----------
    xrdata : xarray.Dataset
        The xarray dataset to be exported.
    filename : str
        The file name for the output TIFF file.
    varnames : list of str, optional
        Names of the variables (bands) to be included in the TIFF file. If not provided, 
        all bands will be included.

    Returns:
    -------
    None
    """
    
    metadata = xrdata.attrs
    
    suffix = (len(filename) + 1)
    
    if filename.endswith('tif'):
        suffix = filename.index('tif')
       

    if len(varnames) > 1:
        metadata['count'] = len(varnames)
        fn = "{}{}.tif".format(filename[:(suffix - 1)], "_".join(varnames))
        xrdata.rio.to_raster(fn)

### TODO: create a class that integrate all plots
class UAVPlots():
    pass

def check_boundarytype(boundsvariable):
    
    """
    Checks and converts the boundary variable to a standard GeoJSON-like dict format.

    Parameters:
    ----------
    boundsvariable : gpd.GeoDataFrame, gpd.GeoSeries, or list of dict
        The boundary variable to be checked and standardized.

    Returns:
    -------
    list of dict
        The boundary variable in a standard format.

    Raises:
    ------
    ValueError
        If the boundary object is not in a recognized format.
    """
    
    if isinstance(boundsvariable, gpd.GeoDataFrame):
        if boundsvariable.shape[0]> 1:
            print('The GeoDataFrame has more than 1 feature; only the first one will be used.')
            
        return [boundsvariable.reset_index(
            ).__geo_interface__['features'][0]['geometry']]
    
    elif isinstance(boundsvariable, gpd.geoseries.GeoSeries):
        return [boundsvariable.reset_index(
            )[0].__geo_interface__['features'][0]['geometry']]
        
    elif isinstance(boundsvariable, list) and all(isinstance(item, dict) for item in boundsvariable):
        return boundsvariable
    
    raise ValueError("The boundary object must be a GeoDataFrame, GeoSeries, or a list of GeoJSON-like dict objects.")
    

class DroneData:
    """
    This class handles UAV imagery using the xarray package and provides methods for data manipulation, 
    analysis, and visualization.

    Attributes:
    ----------
    drone_data : xarray.Dataset
        An xarray dataset representing the UAV image data.
    variable_names: list of str
        A list of strings representing the spectral bands in the UAV image.
    available_vi: dict
        A dictionary of vegetation indices that can be calculated from the spectral imagery.
    """

    def __init__(self, inputpath: str, bands: Optional[List[str]] = None,
                 multiband_image: bool = False, bounds: Optional[dict] = None):
        
        """
        Initialize the DroneData class with the given parameters.

        Parameters:
        ----------
        inputpath : str
            Path where the image data is located.
        bands : list of str, optional
            Names of the spectral bands.
        multiband_image : bool, optional
            Whether the data is a multistack object or composed of separated bands.
        bounds : dict, optional
            GeoDataFrame, GeoSeries, or a list of GeoJSON-like dict to clip the image.
        """
        
        self._bands = ['red', 'green', 'blue'] if bands is None else bands
        self._clusters = np.nan
        self._tiles_pols = None
        self._files_path = self._determine_files_path(inputpath, multiband_image)
        self.bounds_asjson = check_boundarytype(bounds)
        
        if len(self._files_path)>0:
            self.drone_data = self.tif_toxarray(multiband_image, bounds=self.bounds_asjson)
        else:
            raise FileNotFoundError('No file path was found at the specified input path.')
            

    def _determine_files_path(self, inputpath: str, multiband_image: bool) -> List[str]:
        """
        Determine the file paths.

        Parameters:
        ----------
        inputpath : str
            The directory path to search for image files.
        multiband_image : bool
            Whether the image data is in multiband format.

        Returns:
        -------
        List[str]
            A list of file paths for the drone data.

        Raises:
        ------
        ValueError
            If no TIFF files are found in the specified directory.
        """
        if not multiband_image:
            # If the data is not multiband, find individual files for each band.
            return get_files_paths(inputpath, self._bands)
        else:
            # If the data is multiband, expect a single TIFF file.
            try:
                imgfiles = glob.glob(inputpath + "*.tif")[0]
            except:
                raise ValueError(f"No TIFF files found in the directory: {inputpath}")
                    
            return [imgfiles for i in range(len(self._bands))]
                
            

    
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

    def add_layer(self, variable_name,fn = None, imageasarray = None):
        
        self.drone_data = add_2dlayer_toxarrayr(self.drone_data, variable_name,fn = fn, imageasarray = imageasarray)

    def data_astable(self):

        return self.drone_data.to_dataframe()

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

    def tif_toxarray(self, multiband: bool = False, bounds: Optional[dict] = None) -> xarray.Dataset:
        
        """
        Converts TIFF images to an xarray dataset.

        Parameters:
        ----------
        multiband : bool, optional
            Whether the image data is in multiband format.
        bounds : dict, optional
            GeoJSON-like dictionary to clip the image.

        Returns:
        -------
        xarray.Dataset
            The xarray dataset created from the TIFF images.
        """
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
        
        multi_xarray = multi_xarray.rename({'dim_0': 'y', 'dim_1': 'x'})
        multi_xarray = multi_xarray.assign_coords(x=xvalues)
        multi_xarray = multi_xarray.assign_coords(y=yvalues)
        
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

    def to_rgbjpg(self, fn, channels = ['red','green','blue'], newshape = None):
        """_summary_

        Args:
            fn (str): file path
            channels (list, optional): RGB channels names. Defaults to ['red','green','blue'].
            newshape (list, optional): export the image in with a specific size. Default None
        """
         
        fn = fn if fn.endswith('.jpg') else fn+'.jpg'
        
        imgdrs = self.drone_data[channels].to_array().values.copy()
        imgpil = Image.fromarray(np.rollaxis(imgdrs, 0,3).astype(np.uint8))
        
        
        if newshape is not None:
            assert len(newshape) == 2
            imgpil = imgpil.resize(newshape)
        
        imgpil.save(fn)
            
            
        
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
        """
        Function to split an orthomoasaic image into tiles of regular size
        
         Parameters:
        ----------
        width: int
            tile's width size
        height:
            tile's height size
        overlap: float, optional
            value between 0 and 1.0 that sets the overlap percentage between tiles 
        
        """
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

    

