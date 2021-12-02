from numpy.lib.polynomial import poly
import rasterstats as rs
import pandas as pd
import xarray
import numpy as np
import geopandas as gpd

from itertools import product

from rasterio import windows
from shapely.geometry import Polygon

import math

from rasterio.crs import CRS
from rasterio.transform import Affine
from rasterio import features

import affine

from skimage.transform import SimilarityTransform
from skimage.transform import warp
from skimage.registration import phase_cross_correlation


def xy_fromtransform(transform, width, height):

    T0 = transform
    T1 = T0 * affine.Affine.translation(0.5, 0.5)
    rc2xy = lambda r, c:  T1 * (c, r)
    xvals = []
    for i in range(width):
        xvals.append(rc2xy(0, i)[0])

    yvals = []
    for i in range(height):
        yvals.append(rc2xy(i, 0)[1])

    
    xvals = np.array(xvals)
    xvals = np.sort(xvals,axis = 0)[::-1]

    yvals = np.array(yvals)
    yvals = np.sort(yvals,axis = 0)[::-1]

    return [xvals, yvals]

def register_image_shift(data, shift):
    tform = SimilarityTransform(translation=(-shift[1], shift[0]))
    imglist = []
    for i in range(data.shape[2]):
        imglist.append(warp(data[:,:,i], inverse_map = tform, order = 0, preserve_range = True))

    return np.dstack(imglist)


def register_xarray(xarraydata, shift):

    data = xarraydata.copy()
    dataarray = np.dstack([data[i].data for i in list(data.keys())])

    imgregistered = register_image_shift(dataarray,shift)

    for i,band in enumerate(list(data.keys())):
        data[band].values = imgregistered[:,:,i]
    
    return data


def find_shift_between2xarray(offsetdata, refdata, band = 'red', clipboundaries = None, buffer = None):

    if clipboundaries is not None:
        offsetdata = clip_xarraydata(offsetdata, clipboundaries, bands = [band], buffer = buffer)
        refdata = clip_xarraydata(refdata, clipboundaries, bands = [band] , buffer = buffer)

    offsetdata = offsetdata.fillna(0)
    refdata = refdata.fillna(0)

    if refdata[band].shape[0] != offsetdata[band].shape[0]:
        refdata = resample_xarray(refdata,offsetdata).fillna(0)
    
    shift, error, diffphase = phase_cross_correlation(refdata[band].data, offsetdata[band].data)
    
    return shift, error, diffphase


# adapated from https://github.com/Devyanshu/image-split-with-overlap
def start_points(size, split_size, overlap=0):
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



def clip_xarraydata(xarraydata, gpdata, bands = None, buffer = None):

    gpdataclip = gpdata.copy()
    if buffer is not None:
        geom = gpdataclip.geometry.values.buffer(buffer, join_style=2)
    else:
        geom =gpdataclip.geometry.values
    
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
    tr= transform_fromxy(clippedmerged.x.values, clippedmerged.y.values, np.abs(xarraydata.attrs['transform'][0]))
    clippedmerged.attrs['transform'] = tr[0]
    clippedmerged.attrs['crs'] = xarraydata.attrs['crs']
    clippedmerged.attrs['width'] = clippedmerged[list(clippedmerged.keys())[0]].shape[1]
    clippedmerged.attrs['height'] = clippedmerged[list(clippedmerged.keys())[0]].shape[0]

    return clippedmerged




def resample_xarray(xarraydata, xrreference):

    refimagephase = xarraydata.interp(x=xrreference['x'].values, y=xrreference['y'].values)
    refimagephase.attrs['transform'] = transform_fromxy(
        xrreference.x.values, 
        xrreference.y.values, xrreference.attrs['transform'][0])[0]

    refimagephase.attrs['height'] = refimagephase[list(refimagephase.keys())[0]].shape[0]
    refimagephase.attrs['width'] = refimagephase[list(refimagephase.keys())[0]].shape[1]
    refimagephase.attrs['dtype'] = refimagephase[list(refimagephase.keys())[0]].data.dtype

    return refimagephase


def transform_fromxy(x, y, spr):

    gridX,gridY = np.meshgrid(x,y)
    
    return [Affine.translation(gridX[0][0]-spr/2, gridY[0][0]-spr/2)*Affine.scale(spr,spr), gridX.shape]


def transform_frombb(bb, spr):

    xRange = np.arange(bb[0],bb[2]+spr,spr)
    yRange = np.arange(bb[1],bb[3]+spr,spr)
    return transform_fromxy(xRange, yRange, spr)


def rasterize_using_bb(gpdf, bb, crs, sres = 0.01):

    transform, imgsize = transform_frombb(bb, sres)
    rasterCrs = CRS.from_epsg(crs)
    rasterCrs.data

    return(features.rasterize(zip(gpdf.geometry,gpdf.iloc[:,0].values),
                              out_shape=[imgsize[0],imgsize[1]], transform=transform))
                              

def coordinates_fromtransform(transform, imgsize):
        
    # All rows and columns
    cols, rows = np.meshgrid(np.arange(imgsize[0]), np.arange(imgsize[1]))

    # Get affine transform for pixel centres
    T1 = transform * Affine.translation(0.5, 0.5)
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T1

    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = np.vectorize(rc2en, otypes=[np.float64, np.float64])(rows, cols)
    return [eastings, northings]



def list_tif_2xarray(listraster, transform, crs, nodata = 0,bands_names=None):

    
    imgindex = 1
    metadata = {
        'transform' : transform,
        'crs': crs,
        'width': listraster[0].shape[1],
        'height': listraster[1].shape[0]
    }
    if(bands_names is None):
        bands_names = ['band_{}'.format(i) for i in range(len(listraster))]

    riolist = []
    imgindex = 1
    for i in range(len(listraster)):
        img= listraster[i]
        xrimg = xarray.DataArray(img)
        xrimg.name = bands_names[i]
        riolist.append(xrimg)
        imgindex += 1

    # update nodata attribute
    metadata['nodata'] = nodata
    metadata['count'] = imgindex

    multi_xarray = xarray.merge(riolist)
    multi_xarray.attrs = metadata

    ## assign coordinates
    
    x, y = coordinates_fromtransform(transform,
    [listraster[0].shape[1],listraster[0].shape[0]])
    multi_xarray = multi_xarray.rename({'dim_0': 'y', 'dim_1': 'x'})

    multi_xarray = multi_xarray.assign_coords(x=np.sort(np.unique(x)))
    multi_xarray = multi_xarray.assign_coords(y=np.sort(np.unique(y)))

    

    return multi_xarray


def split_xarray_data(xr_data, polygons=True, **kargs):
    """

    :param xr_data: xarray data with x and y coordinates names
    :param polygons: export polygons
    :param kargs:
    :return:
    """
    xarrayall = xr_data.copy()
    m = get_tiles(xarrayall.attrs, **kargs)

    boxes = []
    imgslist = []
    i = 0
    for window, transform in m:

        xrwindowsel = xarrayall.isel(y=window.toslices()[0],
                                     x=window.toslices()[1]).copy()

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
            i += 1
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


def check_border(coord, sz):
    if(coord >= sz):
            coord = sz -1 
    return coord

def from_xyxy_2polygon(x1,y1,x2,y2):
    xpol = [x1, x2,
            x2, x1,
            x1]
    ypol = [y1, y1,
            y2, y2,
            y1]

    return Polygon(list(zip(xpol, ypol)))


def from_bbxarray_2polygon(bb, xarrayimg):
    #pixel_size = imgsdata[id_img].attrs['transform'][0]
    imgsz = xarrayimg.attrs['width']
    xcoords = xarrayimg.coords['x'].values
    ycoords = xarrayimg.coords['y'].values

    bb = bb if isinstance(bb, np.ndarray) else np.array(bb)
    listcoords = []
    for i in bb:
        listcoords.append(check_border(i, imgsz))
    
    x1,y1,x2,y2 =  listcoords

    return from_xyxy_2polygon(xcoords[int(x1)],ycoords[int(y1)],
                              xcoords[int(x2)],ycoords[int(y2)])


def AoI_from_polygons(p1, p2):
    intersecarea = p1.intersection(p2).area
    unionarea = p1.union(p2).area
    return intersecarea/unionarea

def calculate_AoI_fromlist(data, ref, list_ids):

    area_per_iter = []
    ylist  = []
    for y in list_ids:

        ref2 = data.loc[data.id==y,].copy()
        area_per_iter.append(AoI_from_polygons(ref2.geometry.values[0], 
                                                ref))
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

    xlist  = []
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

def merging_overlaped_polygons(polygons, aoi_limit = 0.3, intersec_ratio = 0.75):

    df = polygons.copy()
    listids = df.id.values
    crs_s = polygons.crs

    listdef_pols = []
    while len(listids) > 0:

        rowit = listids[0]
        refpol =  df.loc[df.id==rowit,]
        data_temp1=df.loc[df.id!=rowit,]

        stest = [i for i in range(len(data_temp1['geometry'].values)) 
                if data_temp1['geometry'].values[i].intersects(refpol.geometry.values[0])]
        if len(stest):
            overlaps = data_temp1.id.values[np.array(stest)].tolist()
            overlaps.append(rowit)
        else:
            overlaps = [rowit]

        refpol =  df.loc[df.id==rowit,]
        data_temp1=df.loc[df.id!=rowit,]

        if len(overlaps)>1:

            overlapsc = overlaps.copy()
            overlapsc = [i for i in overlapsc if not i == rowit]

            refpol = df.loc[df.id==rowit,].copy()

            area_per_iter, _ = calculate_AoI_fromlist(df, refpol.geometry.values[0], overlapsc)
                
            todelete = []

            vals = np.array(overlapsc)[(np.array(area_per_iter) > aoi_limit)]
            if len(vals):
                vals = list(vals)
                vals.append(rowit)
                for z in vals:
                    todelete.append(z)

            testlist = []

            for y in np.unique(todelete):
                ref2 = df.loc[df.id==y,].copy()
                testlist.append(ref2)
            
            if len(todelete)==0:
                todelete = [rowit]
                testlist =  [df.loc[df.id==rowit,].copy()]

            p = best_polygon(testlist)
            overlaps2 = [i for i in np.array(overlaps) if not i in todelete]

            scoren = pd.concat(testlist).score.max()
            mergepol = gpd.GeoDataFrame({'pred':refpol.pred.values,
                                        'score': [scoren], 
                                        'geometry': p ,
                                        'tile':refpol.tile.values,
                                        'id': rowit},
                                        
                                    crs=crs_s)

            if len(overlaps2):
                p1 = mergepol.geometry.values[0]

                for z in overlaps2:
                    p2 = df.loc[df.id==z,].copy()
                    intersecarea = p1.intersection(p2.geometry.values[0]).area

                    if (intersecarea/p2.geometry.values[0].area)>intersec_ratio:
                        todelete.append(z)

                    elif (intersecarea/p1.area)>intersec_ratio:
                        p = best_polygon([mergepol, p2])
                        scoren = pd.concat(testlist).score.max()
                        mergepol = gpd.GeoDataFrame({'pred':refpol.pred.values,
                                        'score': [scoren], 
                                        'geometry': p ,
                                        'tile':refpol.tile.values,
                                        'id': rowit},
                                    crs=crs_s)
                        todelete.append(z)

            listdef_pols.append(mergepol)
                    
        else:
            todelete = [rowit]
            listdef_pols.append(refpol)

        df = df[~df['id'].isin(todelete)]

        listids = df.id.values
    
    return listdef_pols



