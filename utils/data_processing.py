import numpy as np
import rasterio.features
import json
from PIL import Image
import os
from pathlib import Path


def assign_valuestoimg(data, height, width, na_indexes=None):
    ids_notnan = np.arange(height *
                           width)

    climg = np.zeros(height *
                     width, dtype='f')

    if na_indexes is not None:
        ids_notnan = np.delete(ids_notnan, na_indexes, axis=0)

    climg[list(ids_notnan)] = data
    #
    return climg.reshape(height, width)


def mask_usingGeometry(shp, xr_data):
    ## converto to json
    jsonshpdict = shp.to_json()
    jsonshp = [feature["geometry"]
               for feature in json.loads(jsonshpdict)["features"]]

    # create mask
    return rasterio.features.geometry_mask(jsonshp,
                                           out_shape=(len(xr_data.y),
                                                      len(xr_data.x)),
                                           transform=xr_data.attrs['transform'],
                                           invert=True)


def get_maskmltfeatures(shp, sr_data, features_list, featurename='id'):
    if isinstance(features_list, list):

        if len(features_list) > 1:
            shpmask = []
            for i in features_list:
                shp_subset = shp[shp[featurename] == i]
                shpmask.append(mask_usingGeometry(shp_subset,
                                                  sr_data))

            shpmaskdef = shpmask[0]
            for i in range(1, len(shpmask)):
                shpmaskdef = np.logical_or(shpmaskdef, shpmask[i])

            return shpmaskdef


def get_maskmltoptions(dataarray, conditions_list):
    boolean_list = np.nan
    if isinstance(conditions_list, list):

        if len(conditions_list) > 1:
            boolean_list = dataarray == conditions_list[0]
            for i in range(1, len(conditions_list)):
                boolean_list = np.logical_or(boolean_list,
                                             (dataarray == conditions_list[i]))
        else:
            print("this function requires more than one filter condition")

    else:
        print('input is not a list')

    return boolean_list


def scaleminmax(values):
    return ((values - np.nanmin(values)) /
            (np.nanmax(values) - np.nanmin(values)))


def scalestd(values):
    return ((values - (np.nanmean(values) - np.nanstd(values) * 2)) /
            ((np.nanmean(values) + np.nanstd(values) * 2) -
             (np.nanmean(values) - np.nanstd(values) * 2)))


def changenodatatonan(data, nodata=0):
    datac = data.copy()
    if len(datac.shape) == 3:
        for i in range(datac.shape[0]):
            datac[i][datac[i] == nodata] = np.nan

    return datac


def get_nan_idsfromarray(nparray):
    ids = []
    for i in range(nparray.shape[0]):
        ids.append(np.argwhere(
            np.isnan(nparray[i])).flatten())

    # ids = list(chain.from_iterable(ids))
    ids = list(np.concatenate(ids).flat)

    return np.unique(ids)


def from_xarray_2array(xrdata, bands):
    data_list = []
    for i in bands:
        banddata = xrdata[i].data
        banddata[banddata == xrdata.attrs['nodata']] = np.nan
        data_list.append(banddata)

    return np.array(data_list)


def from_xarray_2_rgbimage(xarraydata,
                           bands=None,
                           export_as_jpg=False,
                           ouputpath=None):


    if ouputpath is None:
        ouputpath = "image.jpg"
        directory = ""
    else:
        directory = os.path.dirname(ouputpath)

    if bands is None:
        bands = np.array(list(xarraydata.keys()))[0:3]

    data_tile = from_xarray_2array(xarraydata, bands)

    if data_tile.shape[0] == 3:
        data_tile = np.moveaxis(data_tile, 0, -1)

    image = Image.fromarray(data_tile.astype(np.uint8), 'RGB')
    if export_as_jpg:
        Path(directory).mkdir(parents=True, exist_ok=True)

        if not ouputpath.endswith(".jpg"):
            ouputpath = ouputpath + ".jpg"

        image.save(ouputpath)

        print("Image saved: {}".format(ouputpath))

    return image


def from_xarray_to_table(xrdata, nodataval=None,
                         remove_nan=True, features_names=None):
    if features_names is None:
        npdata = np.array([xrdata[i].data
                           for i in list(xrdata.keys())])
    else:
        npdata = np.array([xrdata[i].data
                           for i in features_names])

    if nodataval is not None:
        npdata = changenodatatonan(npdata,
                                   nodataval)

    # reshape to nlayers x nelements
    npdata = npdata.reshape(npdata.shape[0],
                            npdata.shape[1] * npdata.shape[2])

    idsnan = get_nan_idsfromarray(npdata)
    if remove_nan:
        npdata = np.delete(npdata.T, idsnan, axis=0)

    return [npdata, idsnan]


def resize_3dnparray( array,new_size=512):

    if new_size>array.shape[1] and new_size>array.shape[0]:
        resimg = []
        for i in range(array.shape[0]):
            tmp = array[i].copy()
            tmp = np.hstack([tmp, np.zeros([array.shape[1], (new_size-array.shape[2])])])
            resimg.append(np.vstack([tmp, np.zeros([(new_size-array.shape[1]), new_size])]))
    
    else:
        if new_size>array.shape[1]:
            resimg = []
            for i in range(array.shape[0]):
                tmp = array[i].copy()
                resimg.append(np.vstack([tmp, np.zeros([(new_size-array.shape[1]), new_size])]))


        if new_size>array.shape[2]:
            resimg = []
            for i in range(array.shape[0]):
                tmp = array[i].copy()
                resimg.append(np.hstack([tmp, np.zeros([new_size, (new_size-array.shape[2])])]))

    return np.array(resimg)
