from numpy.lib.polynomial import poly
import rasterstats as rs
import pandas as pd
import xarray
from itertools import product

from rasterio import windows
from shapely.geometry import Polygon

import math


# adapted from https://gis.stackexchange.com/questions/285499/how-to-split-multiband-image-into-image-tiles-using-rasterio

def get_tiles(ds, nrows = None, ncols = None, width=None, height=None):
    ncols_img, nrows_img = ds['width'], ds['height']

    if nrows is not None and ncols is not None:
        width = math.ceil(ncols_img/ncols)
        height = math.ceil(nrows_img/nrows)
    
    offsets = product(range(0, ncols_img, width), range(0, nrows_img, height))
    big_window = windows.Window(col_off=0, row_off=0, width=ncols_img, height=nrows_img)
    for col_off, row_off in  offsets:
    
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds['transform'])
        yield window, transform

def split_xarray_data(xr_data, polygons = True, **kargs):
        
    xarrayall = xr_data.copy()
    m = get_tiles(xarrayall.attrs, **kargs)

    boxes = []
    imgslist = []
    i=0
    for window, transform in m:
        
        xrwindowsel = xarrayall.isel(y = window.toslices()[0], 
                                                x = window.toslices()[1]).copy()

        xrwindowsel.attrs['width'] = xrwindowsel.sizes['x']
        xrwindowsel.attrs['height'] = xrwindowsel.sizes['y']
        xrwindowsel.attrs['transform'] = transform
        imgslist.append(xrwindowsel)
        if polygons:
            coords = window.toranges()
            xcoords = coords[1]
            ycoords = coords[0]
            boxes.append((i, Polygon([(xcoords[0], ycoords[0]), (xcoords[1], ycoords[0]), 
            (xcoords[1], ycoords[1]), (xcoords[0], ycoords[1])])))
            i +=1 
    if polygons:
        output = [imgslist, boxes]
    else:
        output = imgslist
    return output

def get_data_perpoints(xrdata, gpdpoints, var_names=None, long=True):
    """

    :param xrdata:
    :param gpdpoints:
    :param var_names:
    :param long:
    :return:
    """

    if var_names is None:
        var_names = list(xrdata.keys())
    listtest = []
    for i in var_names:
        dataextract = rs.zonal_stats(gpdpoints,
                                     xrdata[i].data,
                                     affine=xrdata.attrs['transform'],
                                     # geojson_out=True,
                                     nodata=xrdata.attrs['nodata'],
                                     stats="mean")
        pdextract = pd.DataFrame(dataextract)
        pdextract.columns = [i]
        listtest.append(pdextract)
    pdwide = pd.concat(listtest, axis=1)
    pdwide['id'] = pdwide.index
    if long:
        output = pdwide.melt(id_vars='id')
    else:
        output = pdwide
    return output


