
from ast import Raise
import numpy as np
import tqdm
import pickle
import xarray
from utils.gis_functions import get_tiles
from shapely.geometry import Polygon
import richdem as rd
import os

def calculate_terrain_layers(xr_data, dem_varname = 'z',attrib = 'slope_degrees', name4d = 'date'):
    """
    Function to calculate terrain attributes from dem layer
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
    """
    Function to split the xarray data into tiles of x y y pixels
    
    :param xr_data: xarray data with x and y coordinates names
    :param polygons: export polygons
    :param kargs:
         nrows:
         ncols:
         width:
         height:
        :param overlap: [0.0 - 1]
        :return:
    :return:
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


def get_meanstd_fromlistxarray(xrdatalist, name4d = 'date'):
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
            
        
            datapervar.append(xrdatalist[idpol][varname].to_numpy())

        mean_dict[varname] = np.nanmean(datapervar)
        std_dict[varname] = np.nanstd(datapervar)

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



