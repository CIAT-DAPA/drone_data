from numpy.lib.polynomial import poly
import rasterstats as rs
import pandas as pd
import xarray
import numpy as np
import geopandas as gpd
from PIL import Image, ImageOps

from itertools import product

from rasterio import windows
from shapely.geometry import Polygon

import math

from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio import features
import rioxarray

import affine

from skimage.registration import phase_cross_correlation

from utils.image_functions import phase_convolution,register_image_shift
from utils.image_functions import radial_filter
from utils.image_functions import cartimg_topolar_transform,border_distance_fromgrayimg

from sklearn.impute import KNNImputer

# check it

def scale_255(data):
    return ((data - data.min()) * (1 / (data.max() - data.min()) * 255)).astype('uint8')


def xy_fromtransform(transform, width, height):
    """
    
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
    refimagephase = xarraydata.interp(x=xrreference['x'].values,
                                      y=xrreference['y'].values,
                                      method=method)
    refimagephase.attrs['transform'] = transform_fromxy(
        xrreference.x.values,
        xrreference.y.values, xrreference.attrs['transform'][0])[0]

    refimagephase.attrs['height'] = refimagephase[list(refimagephase.keys())[0]].shape[0]
    refimagephase.attrs['width'] = refimagephase[list(refimagephase.keys())[0]].shape[1]
    refimagephase.attrs['dtype'] = refimagephase[list(refimagephase.keys())[0]].data.dtype

    return refimagephase


def transform_fromxy(x, y, spr):
    if type(spr) is not list:
        sprx = spr
        spry = spr
    else:
        sprx, spry = spr
    gridX, gridY = np.meshgrid(x, y)

    return [Affine.translation(gridX[0][0] - sprx / 2, gridY[0][0] - spry / 2) * Affine.scale(sprx, spry),
            gridX.shape]


def transform_frombb(bb, spr):
    xRange = np.arange(bb[0], bb[2] + spr, spr)
    yRange = np.arange(bb[1], bb[3] + spr, spr)
    return transform_fromxy(xRange, yRange, spr)


def rasterize_using_bb(gpdf, bb, crs, sres=0.01):
    transform, imgsize = transform_frombb(bb, sres)
    rasterCrs = CRS.from_epsg(crs)
    rasterCrs.data

    return (features.rasterize(zip(gpdf.geometry, gpdf.iloc[:, 0].values),
                               out_shape=[imgsize[0], imgsize[1]], transform=transform))


def coordinates_fromtransform(transform, imgsize):
    """
    Create a longitude, latitude meshgrid based on the spatial affine.
    
    """
    # All rows and columns
    cols, rows = np.meshgrid(np.arange(imgsize[0]), np.arange(imgsize[1]))

    # Get affine transform for pixel centres
    T1 = transform * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T1

    # All east and north (there is probably a faster way to do this)
    eastings, northings = np.vectorize(rc2en,
                                       otypes=[np.float64,
                                               np.float64])(rows, cols)
    return [eastings, northings]


def list_tif_2xarray(listraster, transform, crs, nodata=0, bands_names=None):

    metadata = {
        'transform': transform,
        'crs': crs,
        'width': listraster[0].shape[1],
        'height': listraster[0].shape[0]
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

    x, y = coordinates_fromtransform(transform,
                                     [listraster[0].shape[1], listraster[0].shape[0]])
    multi_xarray = multi_xarray.rename({'dim_0': 'y', 'dim_1': 'x'})

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


def split_xarray_data(xr_data, polygons=True, **kargs):
    """

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


def check_border(coord, sz):
    if (coord >= sz):
        coord = sz - 1
    return coord

def from_polygon_2bbox(pol):
    points = list(pol.exterior.coords)
    x_coordinates, y_coordinates = zip(*points)

    return [min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)]


def from_xyxy_2polygon(x1, y1, x2, y2):
    xpol = [x1, x2,
            x2, x1,
            x1]
    ypol = [y1, y1,
            y2, y2,
            y1]

    return Polygon(list(zip(xpol, ypol)))


def from_bbxarray_2polygon(bb, xarrayimg):
    # pixel_size = imgsdata[id_img].attrs['transform'][0]
    imgsz = xarrayimg.attrs['width']
    xcoords = xarrayimg.coords['x'].values
    ycoords = xarrayimg.coords['y'].values

    bb = bb if isinstance(bb, np.ndarray) else np.array(bb)
    listcoords = []
    for i in bb:
        listcoords.append(check_border(i, imgsz))

    x1, y1, x2, y2 = listcoords

    return from_xyxy_2polygon(xcoords[int(x1)], ycoords[int(y1)],
                              xcoords[int(x2)], ycoords[int(y2)])


def AoI_from_polygons(p1, p2):
    intersecarea = p1.intersection(p2).area
    unionarea = p1.union(p2).area
    return intersecarea / unionarea



def calculate_AoI_fromlist(data, ref, list_ids, invert = False):
    area_per_iter = []
    ylist = []
    for y in list_ids:
        ref2 = data.loc[data.id == y,].copy().geometry.values[0]
        if not invert:
            area_per_iter.append(AoI_from_polygons(ref2,
                                               ref))
        else:
            area_per_iter.append(ref2.intersection(ref).area / ref2.area)
        ylist.append(y)

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
def resize_4dxarray(xrdata, new_size = None, flip = False, name4d = 'date'):
    """
    a functions to resize a 4 dimesionals (date, x, y, variables) xrarray data  

    ----------
    Parameters
    xradata : xarray wich contains all information wuth sizes x, y, variables
    new_size: tuple, with new size in x and y
    flip: boolean, flip the image
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
        imgresized.append(rezize_3dxarray(xrdata.isel(date = dateoi).copy(), new_size, flip=flip))

    imgresized = xarray.concat(imgresized, dim=name4d)
    imgresized[name4d] = xrdata[name4d].values
    imgresized.attrs['count'] = len(list(imgresized.keys()))

    return imgresized

def rezize_3dxarray(xrdata, new_size, flip=True):
    """
    a functions to resize a 3 dimesional xrdata 

    ----------
    Parameters
    xradata : xarray wich contains all information wuth sizes x, y, variables
    new_size: tuple, with new size in x and y
    ----------
    Returns
    xarray:
    """
    newx, newy = new_size
    varnames = list(xrdata.keys())
    dims2d = list(xrdata.dims)
    listnpdata = []

    for i in range(len(varnames)):
        image = Image.fromarray(xrdata[varnames[i]].values.copy())
        imageres = image.resize((newx, newy), Image.BILINEAR)
        if flip:
            imageres = ImageOps.flip(imageres)
        listnpdata.append(np.array(imageres))

    newyvalues = expand_1dcoords(xrdata[dims2d[0]].values, newy)
    newxvalues = expand_1dcoords(xrdata[dims2d[1]].values, newx)

    newtr = transform_fromxy(newxvalues, newyvalues, [
        newxvalues[1] - newxvalues[0],
        newyvalues[1] - newyvalues[0]])
    xrdata = list_tif_2xarray(listnpdata, newtr[0],
                              crs=xrdata.attrs['crs'],
                              bands_names=varnames)

    return xrdata


def stack_as4dxarray(xarraylist, 
                     dateslist=None, 
                     sizemethod='max'):
    if type(xarraylist) is not list:
        raise ValueError('Only list xarray are allowed')

    ydim, xdim = list(xarraylist[0].dims.keys())

    coordsvals = [[xarraylist[i].dims[xdim],
                   xarraylist[i].dims[ydim]] for i in range(len(xarraylist))]

    if sizemethod == 'max':
        sizex, sizexy = np.max(coordsvals, axis=0).astype(np.uint)
    elif sizemethod == 'mean':
        sizex, sizexy = np.mean(coordsvals, axis=0).astype(np.uint)

    # transform each multiband xarray to a standar dims size

    xarrayref = rezize_3dxarray(xarraylist[0], [sizex, sizexy])
    listdatesarray = []
    for i in range(len(xarraylist)):
        listdatesarray.append(resample_xarray(xarraylist[i], xarrayref))

    if dateslist is None:
        dateslist = [i for i in range(len(listdatesarray))]

    mltxarray = xarray.concat(listdatesarray, dim='date')
    mltxarray['date'] = dateslist
    mltxarray.attrs['count'] = len(list(mltxarray.keys()))
    return mltxarray


# filter


def filter_3Dxarray_usingradial(xrdata,name4d = 'date', **kargs):
    
    varnames = list(xrdata.keys())
    
    imgfilteredperdate = []
    for i in range(len(xrdata.date)):
        indlayer = xrdata.isel(date = i).copy()
        indfilter =radial_filter(indlayer[varnames[0]].values, **kargs)
        indlayer = indlayer.where(np.logical_not(np.isnan(indfilter)),np.nan)
        imgfilteredperdate.append(indlayer)
    
    if len(imgfilteredperdate)>0:
        #name4d = list(xrdata.dims.keys())[0]

        mltxarray = xarray.concat(imgfilteredperdate, dim=name4d)
        mltxarray[name4d] = xrdata[name4d].values
    else:
        indlayer = xrdata.copy()
        indfilter =radial_filter(indlayer[varnames[0]].values, **kargs)
        mltxarray = indlayer.where(np.logical_not(np.isnan(indfilter)),np.nan)

    return mltxarray


# impute data 
def impute_4dxarray(xrdata, bandstofill=None, nabandmaskname = None,method='knn',nanvalue = None, name4d='date' ,**kargs):

    varnames = list(xrdata.keys())
    if nabandmaskname is not None and nabandmaskname not in varnames:
        raise ValueError('{} is not in the xarray'.format(nabandmaskname))
    if bandstofill is not None:
        if type(bandstofill) is list:
            namesinlist = [i in varnames for i in bandstofill]
            if np.sum(namesinlist) != len(bandstofill):
                raise ValueError('{} is not in the xarray'.format(
                    np.array(bandstofill)[np.logical_not(namesinlist)]))  
    if bandstofill is None:
        bandstofill = varnames

    imgfilled = []
    for dateoi in range(len(xrdata[name4d])):
        
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

    imgfilled = xarray.concat(imgfilled, dim=name4d)
    imgfilled[name4d] = xrdata[name4d].values
    imgfilled.attrs['count'] = len(list(imgfilled.keys()))

    return imgfilled

def xarray_imputation(xrdata, bands = None,namask = None, imputation_method = 'knn', nanvalue = None,**kargs):
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




def centerto_edgedistances_fromxarray(xrdata, anglestep = 2, nathreshhold = 3, confidence = 0.95, refband = None):
    bandnames = list(xrdata.keys())
    if refband is None:
        refband = bandnames[0]
    ## get from the center to the leaf tip
    refimg = xrdata[refband].copy().values
    #distvector, imgvalues = cartimg_topolar_transform(refimg, anglestep=anglestep, nathreshhold = nathreshhold)
    refimg[np.logical_not(np.isnan(refimg))] = 255
    refimg[np.isnan(refimg)] = 0
    refimg = refimg.astype(np.uint8)
    c,r = border_distance_fromgrayimg(refimg)
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