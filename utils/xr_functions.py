
from ast import Raise
import numpy as np
import tqdm
import pickle
import xarray
from shapely.geometry import Polygon
from rasterio import windows

import rasterio
import itertools
import pandas as pd
import os

from .gis_functions import (get_tiles, resize_3dxarray,
                            resample_xarray,
                            clip_xarraydata, 
                            resample_xarray, 
                            register_xarray,
                            find_shift_between2xarray,
                            list_tif_2xarray,
                            crop_using_windowslice)

from .image_functions import radial_filter, remove_smallpixels,transformto_cielab, transformto_hsv

from .decorators import check_output_fn
from .data_processing import data_standarization, minmax_scale
import json




from typing import List, Optional, Union, Dict


### TODO: FutureWarning: The return type of `Dataset.dims` will be changed to return a set of dimension names in future, 
# in order to be more consistent with `DataArray.dims`. To access a mapping from dimension names to lengths, please use `Dataset.sizes`

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    

def crop_xarray_using_mask(maskdata: np.ndarray,
                           xrdata: xarray.Dataset, 
                           min_threshold_mask : float = 0,
                           buffer: int = None) -> xarray.Dataset:
    """
    Crop an xarray dataset using a mask.

    Parameters:
    -----------
    mask_data : np.ndarray
        Array representing the mask.
    xr_data : xr.Dataset
        Xarray dataset to be cropped.
    min_threshold_mask : float, optional
        Minimum threshold value for the mask. Defaults to 0.
    buffer : int, optional
        Buffer value from the mask data to the image border in pixels. Deafault None.
    Returns:
    --------
    xr.Dataset
        Cropped xarray dataset.
    """
    boolmask = maskdata > min_threshold_mask 

    y1, y2 = np.where(boolmask)[0].min(), np.where(boolmask)[0].max()
    x1, x2 = np.where(boolmask)[1].min(), np.where(boolmask)[1].max()
    
    if 'width' in list(xrdata.attrs.keys()) and 'height' in list(xrdata.attrs.keys()):
        ncols_img, nrows_img = xrdata.attrs['width'], xrdata.attrs['height']
    else:
        nrows_img, ncols_img = xrdata[list(xrdata.keys())[0]].values.shape
    
    if buffer:
        y1 = 0 if (y1-buffer)< 0 else y1-buffer
        x1 = 0 if (x1-buffer)< 0 else x1-buffer
        x2 = ncols_img if (x2+buffer)> ncols_img else x2+buffer
        y2 = nrows_img if (y2+buffer)> nrows_img else y2+buffer
        
      
    big_window = windows.Window(col_off=0, row_off=0, 
                                width=ncols_img, height=nrows_img)
        
    crop_window = windows.Window(col_off=x1, row_off=y1, width=abs(x2 - x1),
                            height=abs(y2-y1)).intersection(big_window)
    
    assert 'transform' in list(xrdata.attrs.keys())
    transform = windows.transform(crop_window, xrdata.attrs['transform'])
    
    xrfiltered = crop_using_windowslice(xrdata.copy(), crop_window, transform)
    
    return xrfiltered
    
def from_dict_toxarray(dictdata, dimsformat = 'DCHW'):
    """
    Convert spatial data from a custom dictionary to an xarray dataset.

    Parameters:
    -----------
    dictdata : Dict[str, Any]
        Custom dictionary containing spatial data.
    dimsformat : str, optional
        Format of dimensions in the resulting xarray dataset. Either 'DCHW' or CHW. Defaults to 'DCHW'.

    Returns:
    --------
    xr.Dataset
        Xarray dataset containing the converted spatial data.
    """
    
    import affine
        
    trdata = dictdata['attributes']['transform']
    crsdata = dictdata['attributes']['crs']
    varnames = list(dictdata['variables'].keys())
    listnpdata = get_data_from_dict(dictdata)
    
    # Process transform data
    if type(trdata) is str:
        trdata = trdata.replace('|','')
        trdata = trdata.replace('\n ',',')
        trdata = trdata.replace(' ','')
        trdata = trdata.split(',')
        trdata = [float(i) for i in trdata]
        if trdata[0] == 0.0 or trdata[4] == 0.0:
            pxsize = abs(dictdata['dims']['y'][0] - dictdata['dims']['y'][1])
            trdata[0] = pxsize
            trdata[4] = pxsize
        
    trd = affine.Affine(*trdata)

    datar = list_tif_2xarray(listnpdata, trd,
                                crs=crsdata,
                                bands_names=varnames,
                                dimsformat = dimsformat,
                                dimsvalues = dictdata['dims'])
    
    if 'date' in list(dictdata['dims'].keys()):
        datar = datar.assign_coords(date=np.sort(
            np.unique(dictdata['dims']['date'])))

        
    return datar

def from_xarray_to_dict(xrdata: xarray.Dataset) -> dict:
    """
    Transform spatial xarray data to a custom dictionary.

    Parameters:
    -----------
    xrdata : xr.Dataset
        Input xarray dataset to be transformed.

    Returns:
    --------
    dict
        Custom dictionary containing variables, dimensions, and attributes of the input xarray dataset.
    """
    
    datadict = {
        'variables':{},
        'dims':{},
        'attributes': {}}

    variables = list(xrdata.keys())
    
    for feature in variables:
        datadict['variables'][feature] = xrdata[feature].values

    for dim in xrdata.sizes.keys():
        if dim == 'date':
            datadict['dims'][dim] = np.unique(xrdata[dim])
        else:
            datadict['dims'][dim] = xrdata[dim].values
    
    for attr in xrdata.attrs.keys():
        if attr == 'transform':
            datadict['attributes'][attr] = list(xrdata.attrs[attr])
        else:
            datadict['attributes'][attr] = '{}'.format(xrdata.attrs[attr])
    
    return datadict


def get_data_from_dict(data: Dict[str, Dict[str, np.ndarray]], 
                       onlythesechannels: Optional[List[str]] = None) -> np.ndarray:
    """
    Extracts data for specified channels from a dictionary and converts it into a NumPy array.

    Parameters
    ----------
    data : Dict[str, Dict[str, np.ndarray]]
        A dictionary where the 'variables' key contains another dictionary mapping channel names to their data.
    onlythesechannels : Optional[List[str]], optional
        A list specifying which channels' data to extract. If None, data for all channels is extracted, by default None.

    Returns
    -------
    np.ndarray
        An array containing the data for the specified channels. The array's shape is (N, ...) where N is the number of channels.

    Examples
    --------
    >>> data = {'variables': {'red': np.array([1, 2, 3]), 'green': np.array([4, 5, 6]), 'blue': np.array([7, 8, 9])}}
    >>> get_data_from_dict(data, onlythesechannels=['red', 'blue'])
    array([[1, 2, 3],
        [7, 8, 9]])
    """
        
    dataasarray = []
    channelsnames = list(data['variables'].keys())
    
    if onlythesechannels is not None:
        channelstouse = [i for i in onlythesechannels if i in channelsnames]
    else:
        channelstouse = channelsnames
    for chan in channelstouse:
        dataperchannel = data['variables'][chan] 
        dataasarray.append(dataperchannel)

    return np.array(dataasarray)
    
class CustomXarray(object):
    """A custom class for handling and exporting UAV data using xarray.

    This class allows for exporting UAV data into pickle and/or JSON files
    and includes functionalities for reading and converting xarray datasets.

    Attributes:
        xrdata (xarray.Dataset): Contains the xarray dataset.
        customdict (dict): Custom dictionary containing channel data, dimensional names, and spatial attributes.
    """
    
    def __init__(self, xarraydata: Optional[xarray.Dataset]= None, 
                 file: Optional[str] = None, 
                 customdict: Optional[bool] = False,
                 filesuffix: str = '.pickle',
                 dataformat: str = 'DCHW') -> None:
        """Initializes the CustomXarray class.

        Args:
            xarraydata (xarray.Dataset, optional):
                An xarray dataset to initialize the class.
            file (str, optional):
                Path to a pickle file containing xarray data.
            customdict (bool, optional):
                Indicates if the pickle file is a dictionary or an xarray dataset.
            filesuffix (str, optional):
                Suffix of the file to read. Defaults to '.pickle'.
            dataformat (str, optional):
                Format of the multi-dimensional data. Defaults to 'DCHW', 'CDHW', 'CHWD', 'CHW'.

        Raises:
            ValueError:
                If the provided data is not of type xarray.Dataset when 'xarraydata' is used.

        Examples:
            ### Initializing by loading data from a pickle file
            custom_xarray = CustomXarray(file='/path/to/data.pickle')
        """
        
        self.xrdata = None
        self._customdict = None
        self._arrayorder = dataformat
        
        if xarraydata:
            #assert type(xarraydata) is 
            if not isinstance(xarraydata, xarray.Dataset):
                raise ValueError("Provided 'xarraydata' must be an xarray.Dataset")
        
            self.xrdata = xarraydata
            
        elif file:
            data = self._read_data(path=os.path.dirname(file), 
                                   fn = os.path.basename(file),
                                   suffix=filesuffix)
              
            if customdict:
                self.xrdata = from_dict_toxarray(data, 
                                                 dimsformat = self._arrayorder)
                
            else:
                self.xrdata = data
            
    
    @check_output_fn
    def _export_aspickle(self, path, fn, suffix = '.pickle') -> None:
        """Private method to export data as a pickle file.

        Args:
            path (str): Path to the export directory.
            fn (str): Filename for export.
            suffix (str, optional): File suffix. Defaults to '.pickle'.

        Returns:
            None
        """

        with open(fn, "wb") as f:
            pickle.dump([self._filetoexport], f)
    
    @check_output_fn
    def _export_asjson(self, path, fn, suffix = '.json'):
        """Private method to export data as a JSON file.

        Args:
            path (str): Path to the export directory.
            fn (str): Filename for export.
            suffix (str, optional): File suffix. Defaults to '.json'.

        Returns:
            None
        """
        
        json_object = json.dumps(self._filetoexport, cls = NpEncoder, indent=4)
        with open(fn, "w") as outfile:
            outfile.write(json_object)
    
    @check_output_fn
    def _read_data(self, path, fn, suffix = '.pickle'):
        """Private method to read data from a file.

        Args:
            path (str): Path to the file.
            fn (str): Filename.
            suffix (str, optional): File suffix. Defaults to '.pickle'.

        Returns:
            Any: Data read from the file.
        """
        
        with open(fn,"rb") as f:
            data = pickle.load(f)
        if suffix == '.pickle':
            if type(data) is list:
                data = data[0]
        return data
      
    def export_as_dict(self, path: str, fn: str, asjson: bool = False,**kwargs):
        """Export data as a dictionary, either in pickle or JSON format.

        Args:
            path (str): Path to the export directory.
            fn (str): Filename for export.
            asjson (bool, optional): If True, export as JSON; otherwise, export as pickle.

        Returns:
            None
        """
        
        self._filetoexport = self.custom_dict
        if asjson:
            self._export_asjson(path, fn,suffix = '.json')
            
        else:
            self._export_aspickle(path, fn,suffix = '.pickle', **kwargs)

    def export_as_xarray(self, path: str, fn: str,**kwargs):
        """Export data as an xarray dataset in pickle format.

        Args:
            path (str): Path to the export directory.
            fn (str): Filename for export.

        Returns:
            None
        """
        
        self._filetoexport = self.xrdata
        self._export_aspickle(path, fn,**kwargs)
    
    @property
    def custom_dict(self) -> dict:
        """Get a custom dictionary representation of the xarray dataset.

        Returns:
            dict: Dictionary containing channel data in array format [variables], dimensional names [dims],
            and spatial attributes [attrs].
        """
        
        if self._customdict is None:
            return from_xarray_to_dict(self.xrdata)
        else:
            return self._customdict
    
    @staticmethod
    def to_array(customdict: Optional[dict]=None, onlythesechannels: Optional[List[str]] = None) -> np.ndarray:
        """Static method to convert a custom dictionary to a numpy array.

        Args:
            customdict (dict, optional): Custom dictionary containing the data.
            onlythesechannels (List[str], optional): List of channels to include in the array.

        Returns:
            np.ndarray: Array representation of the data.
        """
        data = get_data_from_dict(customdict, onlythesechannels)
        return data
        


def add_2dlayer_toxarrayr(xarraydata: xarray.Dataset, variable_name: str,fn:str = None, imageasarray:np.ndarray = None) -> xarray.Dataset:
    """
    Add a 2D layer to an existing xarray dataset.

    Parameters:
    -----------
    xarraydata : xarray.Dataset
        Existing xarray dataset.
    variable_name : str
        Name of the variable to be added.
    fn : str, optional
        File path of the image. Either `fn` or `image_as_array` must be provided.
    image_as_array : np.ndarray, optional
        Image data as a numpy array. Either `fn` or `image_as_array` must be provided.

    Returns:
    --------
    xarray.Dataset
        Updated xarray dataset with the added 2D layer.
    """
    #dimsnames = list(xarraydata.sizes.keys())
    #sizexarray = [dict(xarraydata.sizes)[i] for i in dict(xarraydata.sizes)]
    refdimnames = xarraydata.sizes
    if fn is not None:
        with rasterio.open(fn) as src:
            xrimg = xarray.DataArray(src.read(1))

    elif imageasarray is not None:
        if len(imageasarray.shape) == 3:
            imageasarray = imageasarray[:,:,0]
        xrimg = xarray.DataArray(imageasarray)    
        #y_index =[i for i in range(len(sizexarray)) if xrimg.shape[1] == sizexarray[i]][0]
        #x_index = 0 if y_index == 1 else 1
    newdims = {}
    for keyval in xrimg.sizes:
        xrimg.sizes[keyval]
        posdims = [j for j,keyvalref in enumerate(
            refdimnames.keys()) if xrimg.sizes[keyval] == refdimnames[keyvalref]]
        newdims[keyval] = posdims
    # check double same axis sizes
    if len(newdims[list(newdims.keys())[1]]) >1:
        newdims[list(newdims.keys())[1]] = list(refdimnames.keys())[1]
        newdims[list(newdims.keys())[0]] = list(refdimnames.keys())[0]
    else:
        newdims[list(newdims.keys())[1]] = list(refdimnames.keys())[newdims[list(newdims.keys())[1]][0]]
        newdims[list(newdims.keys())[0]] = list(refdimnames.keys())[newdims[list(newdims.keys())[0]][0]]

    xrimg.name = variable_name
    xrimg = xrimg.rename(newdims)

    return xarray.merge([xarraydata, xrimg])



def stack_as4dxarray(xarraylist, 
                     sizemethod='max',
                     axis_name = 'date', 
                     valuesaxis_names = None,
                     new_dimpos = 0,
                     resizeinter_method = 'nearest',
                     long_dimname = 'x',
                     lat_dimname = 'y',
                     resize = True,
                     **kwargs):
    """
    this function is used to stack multiple xarray along a time axis 
    the new xarray value will have dimension {T x C x H x W}

    Parameters:
    ---------
    xarraylist: list
        list of xarray
    sizemethod: str, optional
        each xarray will be resized to a common size, the choosen size will be the maximun value in x and y or the average
        {'max' , 'mean'} default: 'max'
    axis_name: str, optional
        dimension name assigned to the 3 dimensional axis, default 'date' 
    valuesaxis_name: list, optional
        values for the 3 dimensional axis
    resizeinter_method:
        which resize method will be used to interpolate the grid, this uses cv2
         ({"bilinear", "nearest", "bicubic"}, default: "nearest")
        long_dimname: str, optional
        name longitude axis, default = 'x'
    lat_dimname: str, optional
        name latitude axis, default = 'y'
    
    Return:
    ----------
    xarray of dimensions {T x C x H x W}

    """
    if type(xarraylist) is not list:
        raise ValueError('Only list xarray are allowed')

    ydim = [i for i in list(xarraylist[0].sizes.keys()) if lat_dimname in i][0]
    xdim = [i for i in list(xarraylist[0].sizes.keys()) if long_dimname in i][0]

    coordsvals = [[xarraylist[i].sizes[xdim],
                   xarraylist[i].sizes[ydim]] for i in range(len(xarraylist))]

    if resize:
        if sizemethod == 'max':
            sizex, sizexy = np.max(coordsvals, axis=0).astype(np.uint)
        elif sizemethod == 'mean':
            sizex, sizexy = np.mean(coordsvals, axis=0).astype(np.uint)

        # transform each multiband xarray to a standar dims size

        xarrayref = resize_3dxarray(xarraylist[0], [sizex, sizexy], interpolation=resizeinter_method, blur = False,**kwargs)
    else:
        xarrayref = xarraylist[0].copy()
        
    xarrayref = xarrayref.assign_coords({axis_name : valuesaxis_names[0]})
    xarrayref = xarrayref.expand_dims(dim = {axis_name:1}, axis = new_dimpos)

    resmethod = 'linear' if resizeinter_method != 'nearest' else resizeinter_method

    xarrayref = adding_newxarray(xarrayref, 
                     xarraylist[1:],
                     valuesaxis_names=valuesaxis_names[1:], resample_method = resmethod)

    xarrayref.attrs['count'] = len(list(xarrayref.keys()))
    
    return xarrayref


def adding_newxarray(xarray_ref,
                     new_xarray,
                     axis_name = 'date',
                     valuesaxis_names = None,
                     resample_method = 'nearest'):
    """
    function to add new data to a previous multitemporal imagery
    
    Parameters
    ----------
    xarray_ref : xarray.core.dataset.Dataset
        multitemporal data that will be used as reference

    new_xarray: xarray.core.dataset.Dataset
        new 2d data that will be added to xarray used as reference
    axis_name: str, optional
        dimension name assigned to the 3 dimensional axis, default 'date' 
    valuesaxis_name: list, optional
        values for the 3 dimensional axis

    Return
    ----------
    xarray of 3 dimensions
    
    """
    if type(xarray_ref) is not xarray.core.dataset.Dataset:
        raise ValueError('Only xarray is allowed')
    
    new_xarray = new_xarray if type(new_xarray) is list else [new_xarray]
    if valuesaxis_names is None:
        valuesaxis_names = [i for i in range(len(new_xarray))]
    
    # find axis position
    dimpos = [i for i,dim in enumerate(xarray_ref.sizes.keys()) if dim == axis_name][0]

    # transform each multiband xarray to a standar dims size
    singlexarrayref = xarray_ref.isel({axis_name:0})
    listdatesarray = []
    for i in range(len(new_xarray)):
        arrayresizzed = resample_xarray(new_xarray[i], singlexarrayref, method = resample_method)

        arrayresizzed = arrayresizzed.assign_coords({axis_name : valuesaxis_names[i]})
        arrayresizzed = arrayresizzed.expand_dims(dim = {axis_name:1}, axis = dimpos)
        listdatesarray.append(arrayresizzed)

    mltxarray = xarray.concat(listdatesarray, dim=axis_name)    
    # concatenate with previous results
    xarrayupdated = xarray.concat([xarray_ref,mltxarray], dim=axis_name)
    xarrayupdated.attrs['count'] = len(list(xarrayupdated.keys()))
    return xarrayupdated


def calculate_terrain_layers(xr_data, dem_varname = 'z',attrib = 'slope_degrees', name4d = 'date'):
    import richdem as rd
    
    """Function to calculate terrain attributes from dem layer

    xr_data: 
    
    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    if dem_varname not in list(xr_data.keys()):
        raise ValueError('there is not variable called {dem_varname} in the xarray')
    
    terrainattrslist = []
    #name4d = list(xrdata.dims.keys())[0]
    if len(xr_data.sizes.keys())>2:
        for dateoi in range(len(xr_data[name4d])):
            datadem = xr_data[dem_varname].isel({name4d:dateoi}).copy()
            datadem = rd.rdarray(datadem, no_data=0)
            terrvalues = rd.TerrainAttribute(datadem,attrib= attrib)
            terrainattrslist.append(terrvalues)
                    
        xrimg = xarray.DataArray(terrainattrslist)
        vars = list(xr_data.sizes.keys())
        
        vars = [vars[i] for i in range(len(vars)) if i != vars.index(name4d)]

        xrimg.name = attrib
        xrimg = xrimg.rename({'dim_0': name4d, 
                            'dim_1': vars[0],
                            'dim_2': vars[1]})
    else:
        datadem = xarray.DataArray(xr_data[dem_varname].copy())
        datadem = rd.rdarray(datadem, no_data=0)
        terrvalues = rd.TerrainAttribute(datadem,attrib= attrib)

        vars = list(xr_data.sizes.keys())
        xrimg.name = attrib
        xrimg = xrimg.rename({'dim_0': vars[0], 
                            'dim_1': vars[1]})

    return xr_data.merge(xrimg)


def split_xarray_data(xr_data, polygons=True, **kargs):
    """Function to split the xarray data into tiles of x y y pixels

    Args:
        xr_data (xarray): data cube
        polygons (bool, optional): return polygons. Defaults to True.

    Returns:
        list: list of tiles in xarray format and the spatial boundaries or polygons
    """
    xarrayall = xr_data.copy()
    m = get_tiles(xarrayall.attrs, **kargs)

    boxes = []
    orgnizedwidnowslist = []
    i = 0
    for window, transform in m:

        #xrmasked = crop_using_windowslice(xarrayall, window, transform)
        #imgslist.append(xrmasked)
        orgnizedwidnowslist.append((window,transform))
        if polygons:
            coords = window.toranges()
            xcoords = coords[1]
            ycoords = coords[0]
            boxes.append((i, Polygon([(xcoords[0], ycoords[0]), (xcoords[1], ycoords[0]),
                                      (xcoords[1], ycoords[1]), (xcoords[0], ycoords[1])])))
        i += 1
    if polygons:
        output = [m, boxes]
    #else:
    #    output = imgslist

    else:
        output = orgnizedwidnowslist
    return output


def get_xyshapes_from_picklelistxarray(fns_list, 
                                      name4d = 'date', 
                                      dateref = 26):
    """
    get nin max values from a list of xarray

    ----------
    Parameters
    fns_list : list of pickle filenames
    ----------
    Returns
    xshapes: a list that contains the x sizes
    yshapes: a list that  contains the y sizes
    """
    if not (type(fns_list) == list):
        raise ValueError('fns_list must be a list of pickle')


    xshapes = []
    yshapes = []

    for idpol in tqdm.tqdm(range(len(fns_list))):
        with open(fns_list[idpol],"rb") as fn:
            xrdata = pickle.load(fn)
            
        if len(xrdata.sizes.keys())>2:
            if dateref is not None:
                xrdata = xrdata.isel({name4d:dateref})
        xshapes.append(xrdata.sizes[list(xrdata.sizes.keys())[1]])
        yshapes.append(xrdata.sizes[list(xrdata.sizes.keys())[0]])

    return xshapes, yshapes


def get_minmax_from_picklelistxarray(fns_list, 
                                      name4d = 'date', 
                                      bands = None, 
                                      dateref = 26):
    """
    get nin max values from a list of xarray

    ----------
    Parameters
    fns_list : list of pickle filenames
    ----------
    Returns
    min_dict: a dictionary which contains the minimum values per band
    max_dict: a dictionary which contains the maximum values per band
    """
    if not (type(fns_list) == list):
        raise ValueError('fns_list must be a list of pickle')

    with open(fns_list[0],"rb") as fn:
        xrdata = pickle.load(fn)

    if bands is None:
        bands = list(xrdata.keys())
    

    min_dict = dict(zip(bands, [9999]*len(bands)))
    max_dict = dict(zip(bands, [-9999]*len(bands)))

    for idpol in tqdm.tqdm(range(len(fns_list))):
        with open(fns_list[idpol],"rb") as fn:
            xrdata = pickle.load(fn)
            
        if len(xrdata.sizes.keys())>2:
            if dateref is not None:
                xrdata = xrdata.isel({name4d:dateref})
        for varname in list(bands):
            mindict = min_dict[varname]
            maxdict = max_dict[varname]
            minval = np.nanmin(xrdata[varname].values)
            maxval = np.nanmax(xrdata[varname].values)

            max_dict[varname] = maxval if maxdict< maxval else maxdict
            min_dict[varname] = minval if mindict> minval else mindict

    return min_dict, max_dict

def transform_listarrays(values: List[np.ndarray], 
                         var_channels: Optional[List[int]] = None, 
                         scaler: Dict[str, List[float]] = None, 
                         scale_type: str = 'standardization') -> Dict[str, np.ndarray]:
    """
    Applies scaling to a list of numpy arrays based on the specified scaling type.
    
    Parameters
    ----------
    values : List[np.ndarray]
        A list of numpy arrays to be scaled.
    var_channels : Optional[List[int]], optional
        A list of channel indices to be scaled. If None, all channels are scaled, by default None.
    scaler : Optional[Dict[int, List[float]]], optional
        A dictionary with pre-computed scaling parameters for each channel. 
        If None, the scaler is computed based on `scale_type`, by default None.
    scale_type : str, optional
        The type of scaling to apply. Options are 'standardization' and 'normalization', 
        by default 'standardization'.
    
    Returns
    -------
    List[np.ndarray]
        The list of scaled numpy arrays.
    
    Raises
    ------
    ValueError
        If an unsupported `scale_type` is provided.
    """
    if var_channels is None:
        var_channels = list(range(len(values)))
    if scale_type == 'standardization':
        if scaler is None:
            scaler = {chan:[np.nanmean(values[i]),
                            np.nanstd(values[i])] for i, chan in enumerate(var_channels)}
        fun = data_standarization
    elif scale_type == 'normalization':
        if scaler is None:
            scaler = {chan:[np.nanmin(values[i]),
                            np.nanmax(values[i])] for i, chan in enumerate(var_channels)}
        fun = minmax_scale
    
    else:
        raise ValueError(f'{scale_type} is not an available option')
    
    valueschan = {}
    for i, channel in enumerate(var_channels):
        if channel in list(scaler.keys()):
            val1, val2 = scaler[channel]
            #msk0 = values[i] == 0
            scaleddata = fun(values[i], val1, val2)
            #scaleddata[msk0] = 0
            valueschan[channel] = scaleddata
    
    return valueschan    

def customdict_transformation(customdict, scaler:Dict[str, List[float]] = None, 
                              scalertype: str = 'standarization'):
    """scale customdict

    Args:
        customdict (dict): custom dict
        scaler (dict): dictionary that contains the scalar values per channel. 
                       e.g. for example to normalize the red channel you will provide min and max values {'red': [1,255]}  
        scalertype (str, optional): string to mention if 'standarization' or 'normalization' is gonna be applied. Defaults to 'standarization'.

    Returns:
        xrarray: xrarraytransformed
    """
    ccdict = customdict.copy()
    varchanels = list(ccdict['variables'].keys())
    values =[ccdict['variables'][i] for i in varchanels]
    trvalues = transform_listarrays(values, var_channels = varchanels, scaler = scaler, scale_type =scalertype)
    for chan in list(trvalues.keys()):
        ccdict['variables'][chan] = trvalues[chan]
    

def xr_data_transformation(xrdata, scaler: Dict[str, List[float]] = None, scalertype:str = 'standarization'):
    """scale xrarrays

    Args:
        xrdata (xrarray): xarray that contains data
        scaler (dict): dictionary that contains the scalar values per channel. 
                       e.g. for example to normalize the red channel you will provide min and max values {'red': [1,255]}  
        scalertype (str, optional): string to mention if 'standarization' or 'normalization' is gonna be applied. Defaults to 'standarization'.

    Returns:
        xrarray: xrarraytransformed
    """
    ccxr = xrdata.copy()
    varchanels = list(ccxr.keys())
    values =[ccxr[i].to_numpy() for i in varchanels]
    trvalues = transform_listarrays(values, var_channels = varchanels, scaler = scaler, scale_type =scalertype)
    for chan in list(trvalues.keys()):
        ccxr[chan].values = trvalues[chan]
    
    return ccxr



def get_meanstd_fromlistxarray(xrdatalist):
    """
    get nin max values from a list of xarray

    ----------
    Parameters
    xrdatalist : list of xarrays
    ----------
    Returns
    min_dict: a dictionary which contains the minimum values per band
    max_dict: a dictionary which contains the maximum values per band
    """
    if not (type(xrdatalist) == list):
        raise ValueError('xrdatalist must be a list of xarray')

    mean_dict = dict(zip(list(xrdatalist[0].keys()), [9999]*len(list(xrdatalist[0].keys()))))
    std_dict = dict(zip(list(xrdatalist[0].keys()), [-9999]*len(list(xrdatalist[0].keys()))))
    for varname in list(xrdatalist[0].keys()):
        
        datapervar = []
        for idpol in range(len(xrdatalist)):
            
            datapervar.append(xrdatalist[idpol][varname].to_numpy().flatten())
        #print(list(itertools.chain.from_iterable(datapervar)))
        mean_dict[varname] = np.nanmean(list(itertools.chain.from_iterable(datapervar)))
        std_dict[varname] = np.nanstd(list(itertools.chain.from_iterable(datapervar)))

    return mean_dict, std_dict



def get_minmax_fromlistxarray(xrdatalist, name4d = 'date'):
    """
    get nin max values from a list of xarray

    ----------
    Parameters
    xrdatalist : list of xarrays
    ----------
    Returns
    min_dict: a dictionary which contains the minimum values per band
    max_dict: a dictionary which contains the maximum values per band
    """
    if not (type(xrdatalist) == list):
        raise ValueError('xrdatalist must be a list of xarray')

    min_dict = dict(zip(list(xrdatalist[0].keys()), [9999]*len(list(xrdatalist[0].keys()))))
    max_dict = dict(zip(list(xrdatalist[0].keys()), [-9999]*len(list(xrdatalist[0].keys()))))

    for idpol in range(len(xrdatalist)):
        for varname in list(xrdatalist[idpol].keys()):
            minval = min_dict[varname]
            maxval = max_dict[varname]
            if len(xrdatalist[idpol].sizes.keys())>2:      
                for i in range(xrdatalist[idpol].sizes[name4d]):
                    refvalue = xrdatalist[idpol][varname].isel({name4d:i}).values
                    if minval>np.nanmin(refvalue):
                        min_dict[varname] = np.nanmin(refvalue)
                        minval = np.nanmin(refvalue)
                    if maxval<np.nanmax(refvalue):
                        max_dict[varname] = np.nanmax(refvalue)
                        maxval = np.nanmax(refvalue)
            else:
                refvalue = xrdatalist[idpol][varname].values
                if minval>np.nanmin(refvalue):
                        min_dict[varname] = np.nanmin(refvalue)
                        minval = np.nanmin(refvalue)
                if maxval<np.nanmax(refvalue):
                        max_dict[varname] = np.nanmax(refvalue)
                        maxval = np.nanmax(refvalue)

    return min_dict, max_dict



def shift_andregister_xarray(xrimage, xrreference, boundary = None):
    """
    function register and displace a xrdata using another xrdata as reference
    
    Parameters
    ----------
    xrimage: xrdataset
        data to be regeistered
    xrreference: xrdataset
        data sed as reference to register for resize and displacement
    boundary: shapely, optional
        spatial polygon that will be used to clip both datasets

    """

    shiftconv= find_shift_between2xarray(xrimage, xrreference)

    msregistered = register_xarray(xrimage, shiftconv)

    if boundary is not None:
        msregistered = clip_xarraydata(msregistered,  boundary)
        xrreference = clip_xarraydata(xrreference,  boundary)
        
    msregistered = resample_xarray(msregistered, xrreference)

    return msregistered, xrreference



### filter noise


def filter_3Dxarray_usingradial(xrdata,
                                name4d = 'date', 
                                onlythesedates = None, nanvalue = None,**kargs):
    
    varnames = list(xrdata.keys())

    imgfilteredperdate = []
    for i in range(len(xrdata.date)):
        indlayer = xrdata.isel({name4d:i}).copy()
        if onlythesedates is not None and i in onlythesedates:
            indfilter =radial_filter(indlayer[varnames[0]].values,nanvalue = nanvalue, **kargs)
            if nanvalue is not None:
                
                indlayer = indlayer.where(np.logical_not(indfilter == nanvalue),nanvalue)
            else:
                indlayer = indlayer.where(np.logical_not(np.isnan(indfilter)),np.nan)
        
        elif onlythesedates is None:
            indfilter =radial_filter(indlayer[varnames[0]].values,nanvalue = nanvalue, **kargs)

            if nanvalue is not None:
                indlayer = indlayer.where(np.logical_not(indfilter == nanvalue),nanvalue)
            else:
                indlayer = indlayer.where(np.logical_not(np.isnan(indfilter)),np.nan)
            
        imgfilteredperdate.append(indlayer)
    
    if len(imgfilteredperdate)>0:

        mltxarray = xarray.concat(imgfilteredperdate, dim=name4d)
        mltxarray[name4d] = xrdata[name4d].values
    else:
        indlayer = xrdata.copy()
        indfilter =radial_filter(indlayer[varnames[0]].values,nanvalue = nanvalue, **kargs)
        mltxarray = indlayer.where(np.logical_not(np.isnan(indfilter)),np.nan)

    return mltxarray


def filter_3Dxarray_contourarea(xrdata,
                                name4d = 'date', 
                                onlythesedates = None, **kargs):
    
    varnames = list(xrdata.keys())
    
    imgfilteredperdate = []
    for i in range(len(xrdata[name4d])):
        indlayer = xrdata.isel({name4d:i}).to_array().values.copy()
        if onlythesedates is not None and i in onlythesedates:
            imgmasked =remove_smallpixels(indlayer, **kargs)
            indlayer = list_tif_2xarray(imgmasked, 
                 xrdata.attrs['transform'],
                 crs=xrdata.attrs['crs'],
                 bands_names=list(varnames),
                 nodata=np.nan)
            
        elif onlythesedates is None:
            imgmasked =remove_smallpixels(indlayer, **kargs)
            indlayer = list_tif_2xarray(imgmasked, 
                 xrdata.attrs['transform'],
                 crs=xrdata.attrs['crs'],
                 bands_names=list(varnames),
                 nodata=np.nan)

        imgfilteredperdate.append(indlayer)
    
    if len(imgfilteredperdate)>0:
        #name4d = list(xrdata.dims.keys())[0]

        mltxarray = xarray.concat(imgfilteredperdate, dim=name4d)
        mltxarray[name4d] = xrdata[name4d].values
    else:
        indlayer = xrdata.to_array().values.copy()
        imgmasked =remove_smallpixels(indlayer **kargs)
        mltxarray = list_tif_2xarray(imgmasked, 
                 xrdata.attrs['transform'],
                 crs=xrdata.attrs['crs'],
                 bands_names=list(varnames),
                 nodata=np.nan)

    return mltxarray


def calculate_lab_from_xarray(xrdata: xarray.Dataset, 
                              rgbchannels: List[str] = ['red_ms','green_ms','blue_ms'], 
                              dataformat: str = "CDHW", 
                              deepthdimname: str = 'date') -> xarray.Dataset:
    """ 
    Converts RGB data into Lab color space. For more explanation please click on the following link:
    https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.rgb2lab

    Parameters:
    -----------
        xrdata : xarray.Dataset
            Input RGB data.
        rgbchannels : List[str], optional
            List of channel names representing RGB. Defaults to ['red_ms','green_ms','blue_ms'].
        dataformat : str, optional
            Format of the data. Defaults to "CDHW".
        deepthdimname : str, optional
            Name of the depth dimension. Defaults to 'date'.

    Returns:
        xarray.Dataset: Dataset containing Lab color space data.
    """
    srgb_gamma = {"A": 0.055, "PHI": 12.92, "K0": 0.04045, "GAMMA": 2.4}
    
    refdims = list(xrdata.sizes.keys())
    
    if len(refdims) == 3:
        dpos = dataformat.index('D')
        dpos = dpos if dpos == 0 else dpos - 1
        ndepth = len(xrdata[deepthdimname].values)
        xrdate = []
        for i in range(ndepth):
            
            xrdatad = xrdata.isel({deepthdimname:i})
            xrdepthlist = calculate_lab_from_xarray(xrdatad)
            xrdate.append(xrdepthlist)
        
        xrdate = stack_as4dxarray(xrdate, 
                     axis_name = deepthdimname, 
                     new_dimpos = dpos,
                     valuesaxis_names = xrdata.date.values,
                     resize = False)

    else:
        imgtotr = xrdata[rgbchannels].to_array().values.copy()
        # transform to standard rgb https://en.wikipedia.org/wiki/SRGB.
        # taken from https://github.com/tensorflow/graphics/blob/master/tensorflow_graphics/image/color_space/srgb.py#L41-L75
        srgb = np.where(imgtotr <= srgb_gamma['K0'] / srgb_gamma['PHI'], 
                        imgtotr * srgb_gamma['PHI'], 
                        (1 + srgb_gamma['A']) * (imgtotr**(1 / srgb_gamma['GAMMA'])) - srgb_gamma['A'])

        imglab = transformto_cielab(srgb)
        
        xrdate = xrdata.copy()
        
        for labindex, labename in enumerate(['l','a','b']):
            arrimg = imglab[:,:,labindex]
            arrimg[np.isnan(arrimg)] = 0
            
            xrdate = add_2dlayer_toxarrayr(xrdate,variable_name=labename,
                                           imageasarray=arrimg)
            
    return xrdate
            
    return xrdate


def calculate_hsv_from_xarray(xrdata: xarray.Dataset, 
                              rgbchannels: List[str] = ['red_ms','green_ms','blue_ms'], 
                              dataformat: str = "CDHW", 
                              deepthdimname: str = 'date') -> xarray.Dataset:
    """ 
    Converts RGB data into HSV color space.

    Parameters:
    -----------
        xrdata : xarray.Dataset
            Input RGB data.
        rgbchannels : List[str], optional
            List of channel names representing RGB. Defaults to ['red_ms','green_ms','blue_ms'].
        dataformat : str, optional
            Format of the data. Defaults to "CDHW".
        deepthdimname : str, optional
            Name of the depth dimension. Defaults to 'date'.

    Returns:
        xarray.Dataset: Dataset containing HSV color space data.
    """

    refdims = list(xrdata.sizes.keys())
    
    if len(refdims) == 3:
        dpos = dataformat.index('D')
        dpos = dpos if dpos == 0 else dpos - 1
        ndepth = len(xrdata[deepthdimname].values)
        xrdate = []
        for i in range(ndepth):
            
            xrdatad = xrdata.isel({deepthdimname:i})
            xrdepthlist = calculate_hsv_from_xarray(xrdatad)
            xrdate.append(xrdepthlist)
        
        xrdate = stack_as4dxarray(xrdate, 
                     axis_name = deepthdimname, 
                     new_dimpos = dpos,
                     valuesaxis_names = xrdata.date.values,
                     resize = False)

    else:
        imgtotr = xrdata[rgbchannels].to_array().values.copy()

        imglab = transformto_hsv(imgtotr)
        xrdate = xrdata.copy()
        
        for labindex, labename in enumerate(['h','s','v']):
            arrimg = imglab[:,:,labindex]
            arrimg[np.isnan(arrimg)] = 0
            
            xrdate = add_2dlayer_toxarrayr(xrdate,variable_name=labename,
                                           imageasarray=arrimg)
            
    return xrdate

class XRColorSpace(object):
    """
    A class for converting RGB data within an xarray.Dataset to specified color space values (CIE LAB or HSV).

    Parameters
    ----------
    color_space : str, optional
        The target color space for conversion. Supported values are "cielab" and "hsv". Defaults to "cielab".
    
    Attributes
    ----------
    rgb_channels : List[str]
        The RGB channels to be used for color space conversion.
    xrdata : xarray.Dataset
        The dataset to be transformed.
    color_space : str
        The target color space for the conversion.
    _fun : Callable
        The function to be used for the conversion based on the specified color space.
    
    Raises
    ------
    ValueError
        If an unsupported color space is specified.

    Methods
    -------
    transform(update_data=True)
        Applies the color space transformation to the dataset and optionally updates the dataset.
    """
    
    def __init__(self,
                 color_space: str = "cielab") -> None:

        self.color_space = color_space.lower()
        
        if self.color_space == "cielab":
            self._fun = self._calculate_cielab
        elif self.color_space == "hsv":
            self._fun = self._calculate_hsv
        else:
            raise ValueError("Currently, only ['cielab', 'hsv'] are available.")


    def _calculate_hsv(self, update_data = False, **kwargs):
        """
        Calculates the HSV (Hue, Saturation Value) color space values from RGB channels of an xarray dataset.
        """
        xrdatac = self.xrdata.copy()
        xrdatac = calculate_hsv_from_xarray(xrdatac, dataformat=self._array_order, rgbchannels = self.rgb_channels, **kwargs)

        return xrdatac
        
    
    def _calculate_cielab(self, update_data = False, **kwargs):

        """
        Calculates the CIE LAB color space values from RGB channels of an xarray dataset.
        """
        xrdatac = self.xrdata.copy()
        xrdatac = calculate_lab_from_xarray(xrdatac, dataformat=self._array_order, rgbchannels = self.rgb_channels, **kwargs)
                    
            
        return xrdatac
    
    def transform(self, xrdata: xarray.Dataset, rgb_channels: List[str], array_order: str = "CHW") -> xarray.Dataset:
        """
        Applies the specified color space transformation to the dataset.

        Parameters
        ----------
        xrdata : xarray.Dataset
            The xarray dataset that contains the RGB channels.
        rgb_channels : List[str]
            List of channel names representing RGB, e.g., ['red', 'green', 'blue'].
        array_order : str, optional
            The order of array dimensions. Defaults to "CHW".

        update_data : bool, optional
            If True, updates the `xrdata` attribute with the transformed dataset. Defaults to True.

        Returns
        -------
        Any
            The transformed xarray dataset.
        """
        self.xrdata = xrdata
        self.rgb_channels = rgb_channels
        self._array_order = array_order

        return self._fun()
