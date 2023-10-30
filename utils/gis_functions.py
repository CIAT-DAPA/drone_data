from numpy.lib.polynomial import poly

import pandas as pd
import xarray
import numpy as np
import geopandas as gpd
from PIL import Image, ImageOps
import pickle
from itertools import product

from rasterio import windows
from shapely.geometry import Polygon

import math

from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio import features
import rioxarray
import cv2
import affine

from skimage.registration import phase_cross_correlation
from sklearn.impute import KNNImputer

from .image_functions import phase_convolution,register_image_shift

from .image_functions import getcenter_from_hull,border_distance_fromgrayimg
from .image_functions import hist_3dimg
    

# check it

def scale_255(data):
    return ((data - data.min()) * (1 / (data.max() - data.min()) * 255)).astype('uint8')

## basic
def xy_fromtransform(transform, width, height):
    """
    this function is for create longitude and latitude range values from the 
    spatial transformation matrix

    Parameters:
    ----------
    transform: list
        spatial tranform matrix
    width: int
        width size
    height: int
        heigth size

    Returns:
    ----------
    two list, range of unique values for x and y
    """
    T0 = transform
    T1 = T0 * affine.Affine.translation(0.5, 0.5)
    rc2xy = lambda r, c: T1 * (c, r)
    xvals = []
    for i in range(width):
        xvals.append(rc2xy(0, i)[0])

    yvals = []
    for i in range(height):
        yvals.append(rc2xy(i, 0)[1])

    xvals = np.array(xvals)
    if transform[0] < 0:
        xvals = np.sort(xvals, axis=0)[::-1]

    yvals = np.array(yvals)
    if transform[4] < 0:
        yvals = np.sort(yvals, axis=0)[::-1]

    return [xvals, yvals]


def register_xarray(xarraydata, shift):
    data = xarraydata.copy()
    dataarray = np.dstack([data[i].data for i in list(data.keys())])

    imgregistered = register_image_shift(dataarray, shift)

    for i, band in enumerate(list(data.keys())):
        data[band].values = imgregistered[:, :, i]

    return data


def find_shift_between2xarray(offsetdata, refdata, band='red', clipboundaries=None, buffer=None,
                              method="convolution"):
    """ 
    This function will find an image displacement with respect to a reference image
    The method apply a fourier transformation and find the cross correlation between both images

    Parameters:
    ----------
    offsetdata: xarray data
        This cube contains the information that is offset
    refdata: xarray data
        contains the image that will be used as referenced
    band: str
        the channel name that will be used as reference
    clipboundaries: geopandas geometry, optional
        A geometry that will help to analize a region in specific instead of all image
    buffer: float, optional
        if a buffer will be aplied to the mask geometry, this value is in meters.

    Returns
    -------
    tuple with the displacement in x and y axis     
    """
    if clipboundaries is not None:
        offsetdata = clip_xarraydata(offsetdata, clipboundaries, bands=[band], buffer=buffer)
        refdata = clip_xarraydata(refdata, clipboundaries, bands=[band], buffer=buffer)

    offsetdata = offsetdata.fillna(0)
    refdata = refdata.fillna(0)

    if refdata[band].shape[0] != offsetdata[band].shape[0]:
        refdata = resample_xarray(refdata, offsetdata).fillna(0)

    refdata = np.expand_dims(refdata[band].data, axis=2)

    offsetdata = np.expand_dims(scale_255(offsetdata[band].data), axis=2)

    if method == "convolution":
        shift = phase_convolution(refdata, offsetdata)
    if method == "cross_correlation":
        shift, _, _ = phase_cross_correlation(refdata, offsetdata)

    shiftt = shift.copy()

    if (shift[0]) < 0 and (shift[1]) >= 0:
        shiftt[0] = -1 * shiftt[0]
        shiftt[1] = -1 * shiftt[1]

    if (shift[0]) < 0 and (shift[1]) < 0:
        shiftt[0] = -1 * shiftt[0]
        shiftt[1] = -1 * shiftt[1]

    if (shift[0]) >= 0 and (shift[1]) >= 0:
        shiftt[1] = -1 * shiftt[1]
        shiftt[0] = -1 * shiftt[0]

    # if(shift[1])<0 and (shift[0])== 0:
    #    shiftt[1] = -1*shiftt[1]

    if (shift[1]) < 0 and (shift[0]) >= 0:
        shiftt[1] = -1 * shiftt[1]
        shiftt[0] = -1 * shiftt[0]

    shiftt = [shiftt[0], shiftt[1]]

    return [shiftt[0], shiftt[1]]


# adapated from https://github.com/Devyanshu/image-split-with-overlap
def start_points(size, split_size, overlap=0.0):
    """
    this functions initalize the tile in a corner
    """
    points = [0]
    stride = int(split_size * (1 - overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(pt)
            break
        else:
            points.append(pt)
        counter += 1
    return points


# adapted from
# https://gis.stackexchange.com/questions/285499/how-to-split-multiband-image-into-image-tiles-using-rasterio

def get_windows_from_polygon(ds, polygon):
    """

    :param ds: raster metadata
    :param polygon: feature geometry
    
    """

    bbox = from_polygon_2bbox(polygon)
    window_ref = windows.from_bounds(*bbox,ds['transform'])
    transform = windows.transform(window_ref, ds['transform'])

    return [window_ref, transform]

def get_tiles(ds, nrows=None, ncols=None, width=None, height=None, overlap=0.0):
    """

    :param ds: raster metadata
    :param nrows:
    :param ncols:
    :param width:
    :param height:
    :param overlap: [0.0 - 1]
    :return:
    """
    # get width and height from xarray attributes
    ncols_img, nrows_img = ds['width'], ds['height']

    if nrows is not None and ncols is not None:
        width = math.ceil(ncols_img / ncols)
        height = math.ceil(nrows_img / nrows)

    # offsets = product(range(0, ncols_img, width), range(0, nrows_img, height))

    col_off_list = start_points(ncols_img, width, overlap)
    row_off_list = start_points(nrows_img, height, overlap)

    offsets = product(col_off_list, row_off_list)
    big_window = windows.Window(col_off=0, row_off=0, width=ncols_img, height=nrows_img)
    for col_off, row_off in offsets:
        window = windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds['transform'])
        yield window, transform


def clip_xarraydata(xarraydata, gpdata, bands=None, buffer=None):
    gpdataclip = gpdata.copy()
    if buffer is not None:
        geom = gpdataclip.geometry.values.buffer(buffer, join_style=2)
    else:
        geom = gpdataclip.geometry.values

    listclipped = []
    if bands is None:
        bands = list(xarraydata.keys())

    for band in bands:
        xrtoclip = xarraydata[band].copy()
        xrtoclip = xrtoclip.rio.set_crs(xarraydata.attrs['crs'])
        listclipped.append(xrtoclip.rio.clip(geom, gpdataclip.crs))
        clippedmerged = xarray.merge(listclipped)

    clippedmerged.attrs = xarraydata.attrs
    clippedmerged.attrs['nodata'] = xarraydata.attrs['nodata']
    tr = transform_fromxy(clippedmerged.x.values, clippedmerged.y.values,
                          np.abs(xarraydata.attrs['transform'][0]))
    clippedmerged.attrs['transform'] = tr[0]
    clippedmerged.attrs['crs'] = xarraydata.attrs['crs']
    clippedmerged.attrs['width'] = clippedmerged[list(clippedmerged.keys())[0]].shape[1]
    clippedmerged.attrs['height'] = clippedmerged[list(clippedmerged.keys())[0]].shape[0]

    return clippedmerged


def resample_xarray(xarraydata, xrreference, method='linear'):
    """
    Function to resize an xarray data and update its attributes based on another xarray reference 
    this script is based on the xarray's interp() function

    Parameters:
    -------
    xarraydata: xarray
        contains the xarray that will be resized
    xrreference: xarray
        contains the dims that will be used as reference
    method: str
        method that will be used for interpolation
        ({"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial"}, default: "linear")
    
    Returns:
    -------
    a xarray data with new dimensions
    """

    xrresampled = xarraydata.interp(x=xrreference['x'].values,
                                      y=xrreference['y'].values,
                                      method=method)

    xrresampled.attrs['transform'] = transform_fromxy(
        xrreference.x.values,
        xrreference.y.values, xrreference.attrs['transform'][0])[0]

    xrresampled.attrs['height'] = xrresampled[list(xrresampled.keys())[0]].shape[0]
    xrresampled.attrs['width'] = xrresampled[list(xrresampled.keys())[0]].shape[1]
    xrresampled.attrs['dtype'] = xrresampled[list(xrresampled.keys())[0]].data.dtype

    return xrresampled


def transform_fromxy(x, y, spr):
    if type(spr) is not list:
        sprx = spr
        spry = spr
    else:
        sprx, spry = spr
    gridX, gridY = np.meshgrid(x, y)

    return [Affine.translation(
        gridX[0][0] - sprx / 2, gridY[0][0] - spry / 2) * Affine.scale(sprx, spry),
            gridX.shape]


def transform_frombb(bb, spr):
    xRange = np.arange(bb[0], bb[2] + spr, spr)
    yRange = np.arange(bb[1], bb[3] + spr, spr)
    return transform_fromxy(xRange, yRange, spr)


def rasterize_using_bb(values, points_geometry, transform, imgsize):
    """This function create a rasterize vector, given a frame an dataframe values"""
    
    #rasterCrs = CRS.from_epsg(crs)
    #rasterCrs.data

    return (features.rasterize(zip(points_geometry, values),
                               out_shape=[imgsize[0], imgsize[1]], transform=transform))



def coordinates_fromtransform(transform, imgsize):
    """Create a longitude, latitude meshgrid based on the spatial affine.

    Args:
        transform (Affine): Affine matrix transformation
        imgsize (list): height and width 

    Returns:
        _type_: coordinates list in columns and rows
    """
    # All rows and columns
    rows, cols = np.meshgrid(np.arange(imgsize[0]), np.arange(imgsize[1]))

    # Get affine transform for pixel centres
    T1 = transform * Affine.translation(0, 0)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: T1 *(c, r) 

    # All east and north (there is probably a faster way to do this)
    rows, cols = np.vectorize(rc2en,
                                       otypes=[np.float64,
                                               np.float64])(rows, cols)
    return [cols, rows]


def list_tif_2xarray(listraster, transform, crs, nodata=0, 
                     bands_names=None, 
                     dimsformat = 'CHW'):
    """tranform a list of np arrays to a xarray

    Args:
        listraster (list): list of numpy arrays of 3D or 4D dimensions
        transform (Affine): affine transformation matrix
        crs (crs): geograhical coordinate system
        nodata (int, optional): array value for non data. Defaults to 0.
        bands_names (list, optional): list of string with the channels names. Defaults to None.
        dimsformat (list, optional): multi dimensional order. Defaults to CHW.

    Returns:
        xarray dataset: dataset
    """
    
    assert len(listraster)>0
    
    if len(listraster[0].shape) == 2:            
        if dimsformat == 'CHW':
            width = listraster[0].shape[1]
            height = listraster[0].shape[0]
            dims = ['y','x']
        if dimsformat == 'CWH':
            width = listraster[0].shape[0]
            height = listraster[0].shape[1]
            dims = ['y','x']
            
    if len(listraster[0].shape) == 3:
        
        ##TODO: allow multiple formats+
        if dimsformat == 'CDWH':
            width = listraster[0].shape[1]
            height = listraster[0].shape[2]
            dims = ['date','y','x']
            
        if dimsformat == 'CDHW':
            width = listraster[0].shape[2]
            height = listraster[0].shape[1]
            dims = ['date','y','x']
            
        if dimsformat == 'DCHW':
            width = listraster[0].shape[2]
            height = listraster[0].shape[1]
            dims = ['date','y','x']
            
        if dimsformat == 'CHWD':
            width = listraster[0].shape[1]
            height = listraster[0].shape[0]
            dims = ['y','x','date']

    dim_names = {'dim_{}'.format(i):dims[i] for i in range(len(listraster[0].shape))}
    
           
    metadata = {
        'transform': transform,
        'crs': crs,
        'width': width,
        'height': height,
        'count': len(listraster)
        
    }
    if bands_names is None:
        bands_names = ['band_{}'.format(i) for i in range(len(listraster))]

    riolist = []
    imgindex = 1
    for i in range(len(listraster)):
        img = listraster[i]
        xrimg = xarray.DataArray(img)
        xrimg.name = bands_names[i]
        riolist.append(xrimg)
        imgindex += 1

    # update nodata attribute
    metadata['nodata'] = nodata
    metadata['count'] = imgindex

    multi_xarray = xarray.merge(riolist)
    multi_xarray.attrs = metadata

    # assign coordinates
    
    y,x = coordinates_fromtransform(transform,
                                     [height, width])
    
   
    multi_xarray = multi_xarray.rename(dim_names)

    multi_xarray = multi_xarray.assign_coords(x=np.sort(np.unique(x)))
    multi_xarray = multi_xarray.assign_coords(y=np.sort(np.unique(y)))

    return multi_xarray

def crop_using_windowslice(xr_data, window, transform):
    """
    mask a xrarray using a window
    """
    xrwindowsel = xr_data.isel(y=window.toslices()[0],
                                     x=window.toslices()[1]).copy()

    xrwindowsel.attrs['width'] = xrwindowsel.sizes['x']
    xrwindowsel.attrs['height'] = xrwindowsel.sizes['y']
    xrwindowsel.attrs['transform'] = transform

    return xrwindowsel

def get_data_perpoints(xrdata, gpdpoints, var_names=None, long=True):
    import rasterstats as rs
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


def check_border(coord, sz):
    if (coord >= sz):
        coord = sz - 1
    return coord

def from_polygon_2bbox(pol):
    """get min and max coordinates values from a Polygon geometry

    Args:
        pol (Polygon): geometry

    Returns:
        list: (xmin, ymin, xmax, ymax)
    """
    points = list(pol.exterior.coords)
    x_coordinates, y_coordinates = zip(*points)

    return [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]


def from_xyxy_2polygon(x1, y1, x2, y2):
    """transform a coordinates to a Polygon geometry

    Args:
        x1 (int): left or x mininimun coordinate
        y1 (int): top or y min coordinate
        x2 (int): right or x max coordinate
        y2 (int): bottom or y max coordinate

    Returns:
        Polygon: geometry
    """
    xpol = [x1, x2,
            x2, x1,
            x1]
    ypol = [y1, y1,
            y2, y2,
            y1]

    return Polygon(list(zip(xpol, ypol)))


def from_bbxarray_2polygon(bb, xarrayimg):
    """Create a Polygon geometry using the left, top, right and bottom coordiante values of ht eboundary box

    Args:
        bb (_type_): boundary box list values (left, top, right and bottom)
        xarrayimg (xarray): xarray image attributes

    Returns:
        Polygon: geometry
    """
    # pixel_size = imgsdata[id_img].attrs['transform'][0]
    imgsz = xarrayimg.attrs['width']
    xcoords = xarrayimg.coords['x'].values
    ycoords = xarrayimg.coords['y'].values

    bb = bb if isinstance(bb, np.ndarray) else np.array(bb)
    listcoords = []
    for i in bb:
        listcoords.append(check_border(i, imgsz))

    x1, y1, x2, y2 = listcoords
    x2 = len(xcoords)-1 if x2 >= len(xcoords) else x2
    y2 = len(ycoords)-1 if y2 >= len(ycoords) else y2
    
    return from_xyxy_2polygon(xcoords[int(x1)], ycoords[int(y1)],
                              xcoords[int(x2)], ycoords[int(y2)])


def AoI_from_polygons(p1, p2):
    """
    calculate intersection over minimun area
    """
    intersecarea = p1.intersection(p2).area
    unionarea = p1.union(p2).area
    minarea = p2.area 
    if minarea>0:

        if (intersecarea / minarea) > 0.95:
            aoi = 1
        else:
            aoi = intersecarea / unionarea
    else:
        aoi = 0
    return aoi


def calculate_AoI_fromlist(data, ref, list_ids, invert = False):
    area_per_iter = []
    ylist = []
    for y in list_ids:
        ref2 = data.loc[data.id == y,].copy().geometry.values[0]
        if not invert:
            area_per_iter.append(AoI_from_polygons(ref2,
                                               ref))
        else:
            if ref2.area> 0:
                area_per_iter.append(ref2.intersection(ref).area / ref2.area)
            else:
                area_per_iter.append(0)

    return [area_per_iter, ylist]



def get_minmax_pol_coords(p):
    coordsx, coordsy = p.exterior.xy
    x1minr = np.min(coordsx)
    x2minr = np.max(coordsx)
    y1minr = np.min(coordsy)
    y2minr = np.max(coordsy)
    return [[x1minr, x2minr], [y1minr, y2minr]]


def best_polygon(overlaylist):
    xlist = []
    ylist = []
    for i in range(len(overlaylist)):
        x, y = get_minmax_pol_coords(overlaylist[i].geometry.values[0])
        xlist.append(x)
        ylist.append(y)

    x1 = np.min(xlist)
    x2 = np.max(xlist)
    y1 = np.min(ylist)
    y2 = np.max(ylist)

    return from_xyxy_2polygon(x1, y1, x2, y2)


def get_filteredimage(xrdata, channel = 'z',red_perc = 70, refimg = 0, 
                      clip_xarray = False,
                      wrapper = 'hull'):
    """
    Mask a xarray data using percentage

    Parameters:
    ---------

    channel: str, optional
        variable name that will be used as reference
    red_perc: int
        the value that will indicate the final image reduction, 100 will indicate that there won't apply any reduction, 0 will rid off the whole image
    clip_xarray: boolean, optional
        if the value is true, this willl also reduce the image extension, deafault is False
    
    Return:
    --------
    xarray

    """
    if channel is None:
        channel = list(xrdata.keys())[0]
        
    if len(list(xrdata.dims.keys())) >=3:

        vardatename = [i for i in list(xrdata.dims.keys()) 
                        if type(xrdata[i].values[0]) == np.datetime64][0]
        initimageg = xrdata.isel({vardatename:refimg}).copy()
        ydimpos = 1
        xdimpos = 2
    else:
        initimageg = xrdata.copy()
        ydimpos = 0
        xdimpos = 1

    #_,center = centerto_edgedistances_fromxarray(initimageg,  
    #                                            refband = channel,
    #                                            wrapper = wrapper)
        
    if wrapper == 'hull':
        center = getcenter_from_hull(initimageg[channel].values)
        xp,yp = (center[1]),(center[0])
    if wrapper is None:
        center = xrdata[channel].values.shape[xdimpos]//2,xrdata[channel].values.shape[ydimpos]//2
        xp,yp = (center[0]),(center[1])
    
    y = xrdata[channel].values.shape[ydimpos]
    x = xrdata[channel].values.shape[xdimpos]

    pr = red_perc/100

    redy = int(y*pr/2)
    redx = int(x*pr/2)

    lc = int((xp-redx) if (xp-redx) > 0 else 0)
    rc = int((xp+redx) if (x-(xp+redx)) > 0 else x)
    bc = int((yp-redy) if (yp-redy) > 0 else 0)
    tc = int((yp+redy) if (y-(yp+redy)) > 0 else y)
    
    xrfiltered = xrdata.copy()
    for i in list(xrfiltered.keys()):
        npmask = np.zeros(xrdata[i].values.shape)
        if len(list(xrdata.dims.keys())) >=3:    
            npmask[:,bc:tc,lc:rc] = 1
        else:
            npmask[bc:tc,lc:rc] = 1
            
        xrfiltered[i] = xrfiltered[i] * npmask

    if clip_xarray:
        ncols_img, nrows_img = xrfiltered.attrs['width'], xrfiltered.attrs['height']
        big_window = windows.Window(col_off=0, row_off=0, width=ncols_img, height=nrows_img)
    
        window = windows.Window(col_off=lc, row_off=bc, width=abs(lc - rc),
                        height=abs(tc-bc)).intersection(big_window)
        
        transform = windows.transform(window, xrfiltered.attrs['transform'])
        xrfiltered = crop_using_windowslice(xrfiltered.copy(), window, transform)

    return xrfiltered

### merge overlap polygons

def create_intersected_polygon(gpdpols, ref_pol, aoi_treshold = 0.3, invert = False):

    pols_intersected = [gpdpols.id.values[i] for i in range(len(gpdpols['geometry'].values))
            if gpdpols['geometry'].values[i].intersects(ref_pol.geometry.values[0])]
    idstomerge = []
    if len(pols_intersected) > 0:
        
        ### calculate the area of intersection of each polygon
        area_per_iter, _ = calculate_AoI_fromlist(gpdpols, ref_pol.geometry.values[0], pols_intersected,invert)
        

        ### remove those areas with a low instersection area
        idsintersected = np.array(pols_intersected)[(np.array(area_per_iter) > aoi_treshold)]

        idstomerge = list(idsintersected) if len(idsintersected) else []
        ### merge all polygons
        polygonstomerge = [gpdpols.loc[gpdpols.id == y,].copy() for y in np.unique(idstomerge)]
        polygonstomerge.append(ref_pol)

        pol = best_polygon(polygonstomerge)
    else:
        pol = None

        
    return pol, list(np.unique(idstomerge))
    

def merging_overlaped_polygons(polygons, aoi_limit=0.4):

    df = polygons.copy()
    ## create id
    df['id'] = [i for i in range(polygons.shape[0])]
    listids = df.id.values

    listdef_pols = []
    crs_s = polygons.crs
    ## iterate until ther are no more polygons left
    while len(listids) > 0:
        rowit = listids[0]
        refpol = df.loc[df.id == rowit,].copy()
        ## find which polygons are intersected and merged them
        newp, idstodelete = create_intersected_polygon(df, refpol)

        if newp is not None:
    
            idstodelete.append(rowit)
            newgpdpol = gpd.GeoDataFrame({'pred': refpol.pred.values,
                                        'score': [df[df['id'].isin(idstodelete)].score.mean()],
                                        'geometry': newp,
                                        'id': rowit},

                                        crs=crs_s)

            df = df[~df['id'].isin(np.unique(idstodelete))]
            ### find new polygons which intersect the new polygon
            # in this new scenario the intersected area must be bigger
            newarea = aoi_limit *1.65 if aoi_limit *2<1 else 0.9

            newp, idstodelete = create_intersected_polygon(df, newgpdpol, newarea, invert=True)
            if newp is not None:
                
                newgpdpol = gpd.GeoDataFrame({'pred': newgpdpol.pred.values,
                                        'score': newgpdpol.score.values,
                                        'geometry': newp,
                                        'id': rowit},
                                        crs=crs_s)

                df = df[~df['id'].isin(idstodelete)]

        else:
            newgpdpol = refpol
            df = df[~df['id'].isin([rowit])]

        listdef_pols.append(newgpdpol)
        listids = df.id.values
        
    return listdef_pols


def euc_distance(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


def create_linepoints(p1, p2, step=None, npoints=None):
    oriy = 1
    orix = 1
    x1, y1 = p1
    x2, y2 = p2
    if (y2 < y1):
        oriy = -1
    if (x1 > x2):
        orix = -1
    h = euc_distance((x1, y1), (x2, y2))
    alpharad = np.arcsin(np.abs((x2 - x1)) / h)

    listpoints = [(x1, y1)]
    if npoints is not None:
        delta = h / (npoints - 1)
    elif step is not None:
        delta = step
        npoints = int(h / delta)

    deltacum = delta
    while len(listpoints) < npoints:
        xd1 = x1 + ((orix * deltacum) * np.sin(alpharad))
        yd1 = y1 + ((oriy * deltacum) * np.cos(alpharad))

        listpoints.append((xd1, yd1))
        deltacum += delta

    df = pd.DataFrame(listpoints)
    df.columns = ['x', 'y']
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.x, df.y))


def grid_points_usingcorners(topcorner: list, bottomcorner: list, ncolumns=2, nrows=2, idcstart=0, idrstart=0):
    """
    :param topcorner: list coordinates ((x1,y1), (xa2,y2))
    :param bottomcorner: list coordinates ((x1,y1), (xa2,y2))
    """
    t1t2 = create_linepoints(topcorner[0], topcorner[1], npoints=ncolumns)
    b1b2 = create_linepoints(bottomcorner[0], bottomcorner[1], npoints=ncolumns)

    perrows = []
    for a, b in zip(t1t2[['x', 'y']].values, b1b2[['x', 'y']].values):
        perrows.append(create_linepoints((a[0], a[1]), (b[0], b[1]), npoints=nrows).reset_index())

    colvalues = []
    rowvalues = []
    for j in range(ncolumns):
        for i in range(nrows):
            colvalues.append(j + idcstart)
            rowvalues.append(i + idrstart)

    output = pd.concat(perrows)
    output['column'] = colvalues
    output['row'] = rowvalues
    return output


####

def expand_1dcoords(listvalues, newdim):
    delta = np.abs(listvalues[0] - listvalues[-1]) / (newdim - 1)
    if listvalues[0] > listvalues[-1]:
        delta = -1 * delta
    newvalues = [listvalues[0]]

    for i in range(1, (newdim)):
        newval = newvalues[i - 1] + delta

        newvalues.append(newval)

    return np.array(newvalues)


#### reszie single xarray variables
def resize_4dxarray(xrdata, new_size = None, name4d = 'date', **kwargs):
    """
    a functions to resize a 4 dimesionals (date, x, y, variables) xrarray data  

    ----------
    Parameters
    xradata : xarray wich contains all information wuth sizes x, y, variables
    new_size: tuple, with new size in x and y
    
    ----------
    Returns
    xarray:
    """
    if new_size is None:
        raise ValueError("new size must be given")

    #name4d = list(xrdata.dims.keys())[0]
    #ydim, xdim = list(xrdata.dims.keys())[1],list(xrdata.dims.keys())[2]
    imgresized = []
    for dateoi in range(len(xrdata[name4d])):
        imgresized.append(resize_3dxarray(xrdata.isel(date = dateoi).copy(), new_size, **kwargs))

    imgresized = xarray.concat(imgresized, dim=name4d)
    imgresized[name4d] = xrdata[name4d].values
    imgresized.attrs['count'] = len(list(imgresized.keys()))

    return imgresized

def resize_2dimg(image, newx, newy, flip=True, interpolation = 'bilinear', blur= False, kernelsize = (5,5)):
    
    intmethod = cv2.INTER_LINEAR
    if interpolation == 'bilinear':
        intmethod =  cv2.INTER_LINEAR
    if interpolation == 'bicubic':
        intmethod =  cv2.INTER_CUBIC
    if interpolation == 'nearest':
        intmethod = cv2.INTER_NEAREST
    imageres = cv2.resize(image, (newx, newy), interpolation= intmethod)

    if flip:
        imageres = Image.fromarray(imageres)
        imageres = np.array(ImageOps.flip(imageres))
    if blur:
        imageres = cv2.GaussianBlur(imageres,kernelsize,0)

    return imageres

def resize_3dxarray(xrdata, new_size, bands = None,
                    flip=True, 
                    interpolation = 'bilinear', 
                    blur= True, kernelsize = (5,5),
                    long_dimname = 'x',
                    lat_dimname = 'y'):

    """a functions to resize a 3 dimesional xrdata 

    ----------
    Parameters
    xradata : xarray 
        wich contains all information wuth sizes x, y, variables
    new_size: tuple,
        with new size in x and y
    flip: boolean, 
        flip the image
    long_dimname: str, optional
        name longitude axis, default = 'x'
    lat_dimname: str, optional
        name latitude axis, default = 'y'
    ----------
    Returns
    xarray
    """
    # TODO: create an image function for resize

    newx, newy = new_size
    if bands is None:
        varnames = list(xrdata.keys())
    else:
        varnames = bands

    dims2d = list(xrdata.dims)
    ydim = [i for i in list(xrdata.dims.keys()) if lat_dimname in i][0]
    xdim = [i for i in list(xrdata.dims.keys()) if long_dimname in i][0]

    listnpdata = []

    for i in range(len(varnames)):
        #image = Image.fromarray(xrdata[varnames[i]].values.copy())
        #imageres = image.resize((newx, newy), Image.BILINEAR)
        image = xrdata[varnames[i]].values.copy()
        imageres = resize_2dimg(image, newx, newy, 
                                interpolation = interpolation, 
                                flip = flip, blur = blur, kernelsize = kernelsize)

        listnpdata.append(imageres)

    newyvalues = expand_1dcoords(xrdata[ydim].values, newy)
    newxvalues = expand_1dcoords(xrdata[xdim].values, newx)

    newtr = transform_fromxy(newxvalues, newyvalues, [
        newxvalues[1] - newxvalues[0],
        newyvalues[1] - newyvalues[0]])
    xrdata = list_tif_2xarray(listnpdata, newtr[0],
                              crs=xrdata.attrs['crs'],
                              bands_names=varnames)

    return xrdata




# impute data 
def impute_4dxarray(xrdata, bandstofill=None, nabandmaskname = None,method='knn',nanvalue = None, name4d='date',onlythesedates = None ,**kargs):
    """
    fill 3d xarray dimensions (date, x , y)
    ----------
    xrdata : xarray
    bandstofill : str list, optional
        define which bands will be corrected, if this variable is not defined
        the correction will be applied to all available variables
    nabandmaskname : str, optional
        Use a chanel that belongs to the xarray, to be used as a reference in case 
        that some nan values want to be preserved
    method: str, optional
        Currently only k nearest neighbours is available
    name4d: str, optional
        Default name for  the date dimension is date
    onlythesedates: int list, optional
        Apply the filling method to only specific dates.
    Returns
    -------
    xrarray
    """  
    varnames = list(xrdata.keys())
    if nabandmaskname is not None and nabandmaskname not in varnames:
        raise ValueError('{} is not in the xarray'.format(varnames))
    if bandstofill is not None:
        if type(bandstofill) is list:
            namesinlist = [i in varnames for i in bandstofill]
            if np.sum(namesinlist) != len(bandstofill):
                raise ValueError('{} is not in the xarray'.format(
                    np.array(bandstofill)[np.logical_not(namesinlist)]))  
    if bandstofill is None:
        bandstofill = varnames

    imgfilled = []
    if onlythesedates is None:
        datestofill = range(len(list(xrdata[name4d])))
    else:
        datestofill = onlythesedates

    for dateoi in datestofill:
        
        if nabandmaskname is not None:
            filltestref = xrdata.isel({name4d:dateoi})[nabandmaskname].copy()
            mask =np.isnan(filltestref.values)

        else:
            mask = None
        
        xrfilled = xarray_imputation(xrdata.isel({name4d:dateoi}).copy(), 
                                 bands = bandstofill,
                                 namask=mask,imputation_method = method,
                                 nanvalue = nanvalue,**kargs)

        imgfilled.append(xrfilled)
    #def updatexarray_specific_dates(xrdata, onlythesedates, bandstoupdate):
    if onlythesedates is not None:
        for band in bandstofill:
            imgfilledc = []
            count = 0
            for dateoi in range(len(xrdata[name4d].values)):
                if dateoi in onlythesedates:
                    imgfilledc.append(imgfilled[count][band])
                    count +=1
                else:
                    imgfilledc.append(xrdata.isel({name4d:dateoi})[band])
                
            imgfilledc = xarray.concat(imgfilledc, dim=name4d)
            imgfilledc[name4d] = xrdata[name4d]
            xrdata[band] = imgfilledc
            imgfilledc = xrdata

    else:    
        imgfilledc = xarray.concat(imgfilled, dim=name4d)
        imgfilledc[name4d] = xrdata[name4d].values
        imgfilledc.attrs['count'] = len(list(imgfilledc.keys()))

    return imgfilledc

def xarray_imputation(xrdata, bands = None,namask = None, imputation_method = 'knn', nanvalue = None,**kargs):
    """
    this function will fill a 2d array using an imputation method
    ----------
    Parameters
    xrdata : xarray data
        containes the data
    bands : list, optional
        bands or channels that will be filled, if there is not any assigned
        all channels will be processed
    imputation_method:  list, optional
    ----------
    Returns
    xarray : an xarray with all the channels filled using the imputation method
    
    """
    dummylayer = False
    if imputation_method == 'knn':
        impmethod =  KNNImputer(**kargs)

    if bands is None:
        bands = list(xrdata.keys())

    if namask is None:
        imgsize= xrdata[bands[0]].values.shape
        namask = np.ones((imgsize[0]+20,imgsize[1]+20), dtype=bool)*False
        dummylayer = True

    for spband in bands:

        arraytoimput = xrdata[spband].copy().values

        if arraytoimput.shape[0] != namask.shape[0]:
            namaskc = np.zeros(namask.shape)
            namaskc[10:-10,10:-10] = arraytoimput
        else:
            namaskc = arraytoimput
        namaskc[namask] = 0
        namaskc = impmethod.fit_transform(namaskc)
        if dummylayer:
            namaskc[namask] = np.nan
            arraytoimput = namaskc[10:-10,10:-10]
            
        elif arraytoimput.shape[0] != namaskc.shape[0] or arraytoimput.shape[1] != namaskc.shape[1]:
            
            
            difxaxis = arraytoimput.shape[0] - namaskc.shape[0]
            difyaxis = arraytoimput.shape[1] - namaskc.shape[1]
            maxarray = np.empty(arraytoimput.shape)
            maxarray[:,:] = np.nan
            maxarray[:(maxarray.shape[0]-difxaxis),
                     :(maxarray.shape[1]-difyaxis)] = namaskc
            
            namask[maxarray.shape[0]-difxaxis:maxarray.shape[0],
                   (maxarray.shape[1]-difyaxis):maxarray.shape[1]] = True

            maxarray[namask] = np.nan
            arraytoimput = maxarray
        else:
            namaskc[namask] = np.nan
            arraytoimput = namaskc

        if nanvalue is not None:
            arraytoimput[arraytoimput == nanvalue] = np.nan

        xrdata[spband].values  = arraytoimput
    
    return xrdata


def centerto_edgedistances_fromxarray(xrdata, refband = None, 
                                      wrapper = 'hull'):
    bandnames = list(xrdata.keys())
    if refband is None:
        refband = bandnames[0]
    ## get from the center to the leaf tip
    refimg = xrdata[refband].copy().values
    #distvector, imgvalues = cartimg_topolar_transform(refimg, anglestep=anglestep, nathreshhold = nathreshhold)
    refimg[refimg == 0.0] = np.nan
    refimg[np.logical_not(np.isnan(refimg))] = 255
    if np.isnan(refimg).sum() / (refimg.shape[0]*refimg.shape[1]) < .9:
        if wrapper == 'circle':
            refimg[np.isnan(refimg)] = 0
            refimg = refimg.astype(np.uint8)
            c,r = border_distance_fromgrayimg(refimg)
            
        elif wrapper == 'hull':
            c = getcenter_from_hull(refimg, buffernaprc = 15)
            r = None
    else:
        c = [refimg.shape[1]//2,refimg.shape[0]//2]
        r = None
    ## take 
    #borderdistances = []
    #for i in range(len(distvector)):
    #    borderdistances.append(distvector[i][-1:])
    
    #x = np.array(borderdistances).flatten()
    #poslongestdistance = int(np.percentile(x,100*(1-(1-confidence)/2)))

    #mayoraxisref = int(refimg.shape[1]/2) if refimg.shape[1] >= refimg.shape[0] else int(refimg.shape[0]/2)
    #diagnonal = int(mayoraxisref / math.cos(math.radians(45)))
    #mayoraxisref = int(diagnonal) if diagnonal >= mayoraxisref else mayoraxisref
    
    #poslongestdistance = poslongestdistance if poslongestdistance <= mayoraxisref else mayoraxisref

    #return [poslongestdistance, x]
    return [r,c]



def hist_ndxarrayequalization(xrdata, bands = None,
                              keep_original = True,name4d = 'date'):
    """
    change rgb bands contrast by applying an histogram equalization
    ----------
    xrdata : xrarray
    bands : str list, optional
        define which bands will be corrected, if this variable is not defined
        the correction will be applied to all available variables
    Returns
    -------
    xrarray
    """  
    if bands is None:
        bands = list(xrdata.keys())
    ndxarray = xrdata.copy()
    lendims = len(list(ndxarray.dims))

    if lendims == 2:
        ndimgs = hist_3dimg(ndxarray[bands])
    if lendims == 3:
        ndimgs = []
        for j in range(len(bands)):
            barray = ndxarray[bands[j]].to_numpy().copy()
            ndimgs.append(hist_3dimg(barray))
        ndimgs = np.array(ndimgs)
    
    for i in range(len(bands)):
        if keep_original:
            xrimg = xarray.DataArray(ndimgs.astype(float)[i] )
            xrimg.name = bands[i] + '_eq'
            vars = list(ndxarray.dims.keys())
            if lendims == 3:
                vars = [vars[i] for i in range(len(vars)) if i != vars.index(name4d)]
                xrimg = xrimg.rename({'dim_0': name4d, 
                                'dim_1': vars[0],
                                'dim_2': vars[1]})
            else:
                xrimg = xrimg.rename({
                    'dim_1': vars[0],
                    'dim_2': vars[1]})

            ndxarray[bands[i] + '_eq'] = xrimg
            ndxarray.attrs['count'] = len(list(ndxarray.keys()))
        else:
            ndxarray[bands[i]].values = ndimgs.astype(float)[i] 
        
    return ndxarray