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



