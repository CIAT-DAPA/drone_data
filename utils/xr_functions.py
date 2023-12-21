
from ast import Raise
import numpy as np
import tqdm
import pickle
import xarray
from shapely.geometry import Polygon

import rasterio
import itertools
import pandas as pd
import os

from .gis_functions import get_tiles, resize_3dxarray
from .gis_functions import resample_xarray
from .gis_functions import clip_xarraydata, resample_xarray, register_xarray,find_shift_between2xarray
from .gis_functions import list_tif_2xarray

from .image_functions import radial_filter, remove_smallpixels,transformto_cielab

from .decorators import check_output_fn
from .data_processing import data_standarization, minmax_scale
import json

from typing import List, Optional, Union

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
        
def from_dict_toxarray(dictdata, dimsformat = 'DCHW'):
    import affine
        
    trdata = dictdata['attributes']['transform']
    crsdata = dictdata['attributes']['crs']
    varnames = list(dictdata['variables'].keys())
    listnpdata = get_data_from_dict(dictdata)
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
        
    trd = affine.Affine(trdata[0],trdata[1],trdata[2],trdata[3],trdata[4],trdata[5])

    datar = list_tif_2xarray(listnpdata, trd,
                                crs=crsdata,
                                bands_names=varnames,
                                dimsformat = dimsformat)
    
    if 'date' in list(dictdata['dims'].keys()):
        datar = datar.assign_coords(date=np.sort(
            np.unique(dictdata['dims']['date'])))
        
        
    return datar

def from_xarray_to_dict(xrdata):
    """transform spatial xarray data to custom dict

    Args:
        xrdata (xarray): spatial xarray

    Returns:
        dict: qih folowing keys 'variables':chanels information, 'dims':dimensions names, and 'attributes': spatial attributes affine, crs
    """
    
    datadict = {
        'variables':{},
        'dims':{},
        'attributes': {}}

    variables = list(xrdata.keys())
    
    for feature in variables:
        datadict['variables'][feature] = xrdata[feature].values

    for dim in xrdata.dims.keys():
        if dim == 'date':
            datadict['dims'][dim] = np.unique(xrdata[dim])
        
    
    for attr in xrdata.attrs.keys():
        if attr == 'transform':
            datadict['attributes'][attr] = list(xrdata.attrs[attr])
        else:
            datadict['attributes'][attr] = '{}'.format(xrdata.attrs[attr])
    
    return datadict


def get_data_from_dict(data, onlythesechannels = None):
            
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
    """
    A custom class for handling and exporting UAV data using xarray. This class allows for 
    exporting UAV data into pickle and/or JSON files and includes functionalities for reading
    and converting xarray datasets.

    Attributes:
    ----------
    xrdata : xarray.Dataset
        Contains the xarray dataset.
    customdict : dict
        Custom dictionary containing channel data, dimensional names, and spatial attributes.
    """
    
    def __init__(self, xarraydata: Optional[xarray.Dataset]= None, 
                 file: Optional[str] = None, 
                 customdict: Optional[bool] = False,
                 filesuffix: str = '.pickle',
                 dataformat: str = 'DCHW') -> None:
        """
        Initializes the CustomXarray class.

        Parameters:
        ----------
        xarraydata : xarray.Dataset, optional
            An xarray dataset to initialize the class.
        file : str, optional
            Path to a pickle file containing xarray data.
        customdict : bool, optional
            Indicates if the pickle file is a dictionary or an xarray dataset.
        filesuffix : str, optional
            Suffix of the file to read. Defaults to '.pickle'.
        dataformat : str, optional
            Format of the multi-dimensional data. Defaults to 'DCHW', 'CDHW', 'CHWD', 'CHW'.

        Raises:
        ------
        ValueError
            If the provided data is not of type xarray.Dataset when 'xarraydata' is used.
        
        Examples:
        --------
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
                self.xrdata = from_dict_toxarray(data, dimsformat = self._arrayorder)
                
            else:
                self.xrdata = data
            
    
    @check_output_fn
    def _export_aspickle(self, path, fn, suffix = '.pickle') -> None:
        """
        Private method to export data as a pickle file.

        Parameters:
        ----------
        path : str
            Path to the export directory.
        fn : str
            Filename for export.
        suffix : str, optional
            File suffix. Defaults to '.pickle'.

        Returns:
        -------
        None
        """

        with open(fn, "wb") as f:
            pickle.dump([self._filetoexport], f)
    
    @check_output_fn
    def _export_asjson(self, path, fn, suffix = '.json'):
        """
        Private method to export data as a JSON file.

        Parameters:
        ----------
        path : str
            Path to the export directory.
        fn : str
            Filename for export.
        suffix : str, optional
            File suffix. Defaults to '.json'.

        Returns:
        -------
        None
        """
        
        json_object = json.dumps(self._filetoexport, cls = NpEncoder, indent=4)
        with open(fn, "w") as outfile:
            outfile.write(json_object)
    
    @check_output_fn
    def _read_data(self, path, fn, suffix = '.pickle'):
        """
        Private method to read data from a file.

        Parameters:
        ----------
        path : str
            Path to the file.
        fn : str
            Filename.
        suffix : str, optional
            File suffix. Defaults to '.pickle'.

        Returns:
        -------
        Any
            Data read from the file.
        """
        
        with open(fn,"rb") as f:
            data = pickle.load(f)
        if suffix == '.pickle':
            if type(data) is list:
                data = data[0]
        return data
      
    def export_as_dict(self, path: str, fn: str, asjson: bool = False,**kwargs):
        """
        Export data as a dictionary, either in pickle or JSON format.

        Parameters:
        ----------
        path : str
            Path to the export directory.
        fn : str
            Filename for export.
        asjson : bool, optional
            If True, export as JSON; otherwise, export as pickle.

        Returns:
        -------
        None
        """
        
        self._filetoexport = self.custom_dict
        if asjson:
            self._export_asjson(path, fn,suffix = '.json')
            
        else:
            self._export_aspickle(path, fn,suffix = '.pickle', **kwargs)

    def export_as_xarray(self, path: str, fn: str,**kwargs):
        """
        Export data as an xarray dataset in pickle format.

        Parameters:
        ----------
        path : str
            Path to the export directory.
        fn : str
            Filename for export.

        Returns:
        -------
        None
        """
        
        self._filetoexport = self.xrdata
        self._export_aspickle(path, fn,**kwargs)
    
    @property
    def custom_dict(self) -> dict:
        """
        Get a custom dictionary representation of the xarray dataset.

        Returns:
        -------
        dict
            Dictionary containing channel data in array format [variables], dimensional names [dims],
            and spatial attributes [attrs].
        """
        
        if self._customdict is None:
            return from_xarray_to_dict(self.xrdata)
        else:
            return self._customdict
    
    @staticmethod
    def to_array(customdict: Optional[dict]=None, onlythesechannels: Optional[List[str]] = None) -> np.ndarray:
        """
        Static method to convert a custom dictionary to a numpy array.

        Parameters:
        ----------
        customdict : dict, optional
            Custom dictionary containing the data.
        onlythesechannels : List[str], optional
            List of channels to include in the array.

        Returns:
        -------
        np.ndarray
            Array representation of the data.
        """
        
        data = get_data_from_dict(customdict, onlythesechannels)
        return data
        


def add_2dlayer_toxarrayr(xarraydata, variable_name,fn = None, imageasarray = None):

        dimsnames = list(xarraydata.dims.keys())
        if fn is not None:
            with rasterio.open(fn) as src:
                xrimg = xarray.DataArray(src.read(1))
        elif imageasarray is not None:
            if len(imageasarray.shape) == 3:
                imageasarray = imageasarray[:,:,0]

            xrimg = xarray.DataArray(imageasarray)    

        xrimg.name = variable_name
        xrimg = xrimg.rename({'dim_0': dimsnames[0], 'dim_1': dimsnames[1]})

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

    ydim = [i for i in list(xarraylist[0].dims.keys()) if lat_dimname in i][0]
    xdim = [i for i in list(xarraylist[0].dims.keys()) if long_dimname in i][0]

    coordsvals = [[xarraylist[i].dims[xdim],
                   xarraylist[i].dims[ydim]] for i in range(len(xarraylist))]

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
    dimpos = [i for i,dim in enumerate(xarray_ref.dims.keys()) if dim == axis_name][0]

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
    if len(xr_data.dims.keys())>2:
        for dateoi in range(len(xr_data[name4d])):
            datadem = xr_data[dem_varname].isel({name4d:dateoi}).copy()
            datadem = rd.rdarray(datadem, no_data=0)
            terrvalues = rd.TerrainAttribute(datadem,attrib= attrib)
            terrainattrslist.append(terrvalues)
                    
        xrimg = xarray.DataArray(terrainattrslist)
        vars = list(xr_data.dims.keys())
        
        vars = [vars[i] for i in range(len(vars)) if i != vars.index(name4d)]

        xrimg.name = attrib
        xrimg = xrimg.rename({'dim_0': name4d, 
                            'dim_1': vars[0],
                            'dim_2': vars[1]})
    else:
        datadem = xarray.DataArray(xr_data[dem_varname].copy())
        datadem = rd.rdarray(datadem, no_data=0)
        terrvalues = rd.TerrainAttribute(datadem,attrib= attrib)

        vars = list(xr_data.dims.keys())
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
            
        if len(xrdata.dims.keys())>2:
            if dateref is not None:
                xrdata = xrdata.isel({name4d:dateref})
        xshapes.append(xrdata.dims[list(xrdata.dims.keys())[1]])
        yshapes.append(xrdata.dims[list(xrdata.dims.keys())[0]])

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
            
        if len(xrdata.dims.keys())>2:
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

def transform_listarrays(values, varchanels = None, scaler = None, scalertype = 'standarization'):
    
    if varchanels is None:
        varchanels = list(range(len(values)))
    if scalertype == 'standarization':
        if scaler is None:
            scaler = {chan:[np.nanmean(values[i]),
                            np.nanstd(values[i])] for i, chan in enumerate(varchanels)}
        fun = data_standarization
    elif scalertype == 'normalization':
        if scaler is None:
            scaler = {chan:[np.nanmin(values[i]),
                            np.nanmax(values[i])] for i, chan in enumerate(varchanels)}
        fun = minmax_scale
    
    else:
        raise ValueError('{} is not an available option')
    
    valueschan = {}
    for i, channel in enumerate(varchanels):
        if channel in list(scaler.keys()):
            val1, val2 = scaler[channel]
            #msk0 = values[i] == 0
            scaleddata = fun(values[i], val1, val2)
            #scaleddata[msk0] = 0
            valueschan[channel] = scaleddata
    
    return valueschan    

def customdict_transformation(customdict, scaler, scalertype = 'standarization'):
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
    trvalues = transform_listarrays(values, varchanels = varchanels, scaler = scaler, scalertype =scalertype)
    for chan in list(trvalues.keys()):
        ccdict['variables'][chan] = trvalues[chan]
    

def xr_data_transformation(xrdata, scaler = None, scalertype = 'standarization'):
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
    trvalues = transform_listarrays(values, varchanels = varchanels, scaler = scaler, scalertype =scalertype)
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
            if len(xrdatalist[idpol].dims.keys())>2:      
                for i in range(xrdatalist[idpol].dims[name4d]):
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


def calculate_lab_from_xarray(xrdata, rgbchannels = ['red_ms','green_ms','blue_ms'], dataformat = "CDHW", deepthdimname = 'date'):
    """ function to convert RGB data into Lab space color 
    https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.rgb2lab

    Args:
        xrdata (_type_): xarray data
        rgbchannels (list, optional): rgb channels names. Defaults to ['red_ms','green_ms','blue_ms'].
        dataformat (str, optional): what is data xarray format. Defaults to "CDHW".
        deepthdimname (str, optional): if the xarray has three dimensions, the name of the the depth dimension. Defaults to 'date'.

    Returns:
        xarray that include three new channels. L: light A: color from red to green  B: color from yellow to blue
    """
    
    refdims = list(xrdata.dims.keys())
    
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
        imgtotr = xrdata[rgbchannels].to_array().values
        
        imglab = transformto_cielab(imgtotr)
        xrdate = [xrdata]
        for labindex, labename in enumerate(['l','a','b']):
            arrimg = imglab[:,:,labindex]
            arrimg[np.isnan(arrimg)] = 0
            xrimg = xarray.DataArray(arrimg)
            prevdimnames = list(xrimg.dims)
            refdimnames = list(xrdata.dims.keys())
            xrimg.name = labename
            xrimg = xrimg.rename({dname:refdimnames[i] for i,dname in enumerate(prevdimnames)})
            xrdate.append(xrimg)
        xrdate = xarray.merge(xrdate)
            
    return xrdate
    