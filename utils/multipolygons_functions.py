from turtle import addshape
import xarray
from datetime import datetime
import os
import pickle
import numpy as np


from . import drone_data
from .data_processing import find_date_instring
from .gis_functions import clip_xarraydata, resample_xarray, register_xarray,find_shift_between2xarray, stack_as4dxarray
from .xyz_functions import CloudPoints
import geopandas as gpd

from .xyz_functions import get_baseline_altitude
from .gis_functions import impute_4dxarray,xarray_imputation,hist_ndxarrayequalization
from .xyz_functions import calculate_leaf_angle
from .drone_data import calculate_vi_fromxarray


VEGETATION_INDEX = {# rgb bands
'grvi': '(green_ms - red_ms)/(green_ms + red_ms)',
'grvi_eq': '(green_eq - red_eq)/(green_eq + red_eq)',
'grvi_rgb': '(green - red)/(green + red)',
'rgbvi': '((green_ms*green_ms) - (blue_ms*red_ms))/ ((green_ms*green_ms) + (blue_ms*red_ms))',
'rgbvi_eq': '((green_eq*green_eq) - (blue_eq*red_eq))/ ((green_eq*green_eq) + (blue_eq*red_eq))',
'rgbvi_rgb': '((green*green) - (blue*red))/ ((green*green) + (blue*red))',
 # nir indexes
'ndvi': '(nir - red_ms)/(nir + red_ms)',
'ndre': '(nir - edge)/(nir + edge)',
'gndvi': '(nir - green_ms)/(nir + green_ms)',
'regnvi': '(edge - green_ms)/(edge + green_ms)',
'reci': '(nir / edge) - 1',
'negvi': '((nir*nir) - (edge*green_ms))/ ((nir*nir) + (edge*green_ms))',
'savi':  '((nir - red_ms) / (nir + red_ms + 0.5)) * (1.5)'}


def mergemissions_singledate( roi,capturedates = None, 
                              rgb_path = None, ms_path=None, xyz_path=None, 
                              exportrgb =True, 
                              exportxyz = True, 
                              rgb_asreference = True):
    
    
    lenmission = 0   
    suffix = ""
    if rgb_path is not None:
        if capturedates is not None:
            capturedates = [find_date_instring(rgb_path[i]) for i in range(len(rgb_path))]
        if exportrgb:
            suffix +="rgb_"
        
    if ms_path is not None:
        if capturedates is not None:
            capturedates = [find_date_instring(ms_path[i]) for i in range(len(ms_path))]
        suffix +="ms_"
        

    if xyz_path is not None:
        if capturedates is not None:
            capturedates = [find_date_instring(xyz_path[i]) for i in range(len(xyz_path))]
        suffix +="xyz_"
        

    if suffix == "":
        raise ValueError("There are no any mission to process")
    
    
    rgbmsz = mergealldata(roi, lenmission, rgb_path, ms_path, xyz_path, 
                        exportrgb =exportrgb, exportxyz=exportxyz,rgb_asreference = rgb_asreference)


    datesnames = [datetime.strptime(m,'%Y%m%d') for m in capturedates]
    rgbmsz = rgbmsz.expand_dims(dim='date', axis=0)
    rgbmsz['date'] = datesnames
    rgbmsz.attrs['count'] = len(list(rgbmsz.keys()))
        
    return rgbmsz,suffix 


def run_parallel_mergemissions_perpol(j, SHP_PATH, rgb_path = None, ms_path=None, xyz_path=None, output_path=None, 
                        featurename =None, exportrgb =True, exportxyz = True, rgb_asreference = True, preprocess = True):
    cloud_thread = []
    capturedates = None
    allpolygons = gpd.read_file(SHP_PATH)

    lenmission = 0   
    suffix = ""
    if rgb_path is not None:
        capturedates = [find_date_instring(rgb_path[i]) for i in range(len(rgb_path))]
        if exportrgb:
            suffix +="rgb_"
        lenmission = len(rgb_path)
    if ms_path is not None:
        capturedates = [find_date_instring(ms_path[i]) for i in range(len(ms_path))]
        suffix +="ms_"
        lenmission = len(ms_path)

    if xyz_path is not None:
        capturedates = [find_date_instring(xyz_path[i]) for i in range(len(xyz_path))]
        suffix +="xyz_"
        lenmission = len(xyz_path)

    if suffix == "":
        raise ValueError("There are no any mission to process")
    
    for i in range(lenmission):
        cloud_thread.append(mergealldata(allpolygons.loc[j:j,:].copy(),i, rgb_path, ms_path, xyz_path, 
        exportrgb =exportrgb, exportxyz=exportxyz,rgb_asreference = rgb_asreference))


    datesnames = [datetime.strptime(m,'%Y%m%d') for m in capturedates]

    alldata = stack_as4dxarray(cloud_thread, datesnames)
    if featurename is None:
        outfn = os.path.join(output_path, '{}pol_{}_{}_{}.pickle'.format(suffix,j,capturedates[-1],capturedates[0]))
    else:
        idpol = allpolygons.loc[j,featurename]
        outfn = os.path.join(output_path, '{}pol_{}_{}_{}.pickle'.format(suffix,idpol,capturedates[-1],capturedates[0]))
    
    with open(outfn, "wb") as f:
        pickle.dump(alldata, f)
        
    #return alldata 

def fromxyz_file_to_xarray(xyzpaths, gpdpolygon, sres = 0.012,multiprocess= False, nworkers=2):
    """
        This function will create a 2D image based on a 3D cloud points reconstruction
    
    Parameters:
    ----------
    
    xyzpaths: list
        list of filenames with .xyz extension that contained the cloud points information
    gpdpolygon: polygon
        this is a geopandas geometry that is used to indicate which region will be reconstructed 
    sres: float, optional
        this number indicates the meter resolution that will have the image
    multiprocess: boolean, optional
        indicates if the process will be parallized, default is False
    nworkers: integer, optional
        how many cores will be used for paralleizeing the process
    
    Returns:
    ----------
    xarray file with all data plus a new variable called z that conatins the 3D information

    """
    points_perplant = CloudPoints(xyzpaths, gpdpolygon, multiprocess= multiprocess, 
                                  nworkers=nworkers)
    #points_perplant.remove_baseline(method = baselinemethod)

    points_perplant = points_perplant.to_xarray(sp_res= sres)
    #points_perplant = calculate_leaf_angle(points_perplant)
    points_perplant = points_perplant.where(points_perplant.z != 0.)

    return points_perplant
    

def mergealldata(roi,id, rgbfiles = None, msfiles = None, xyzfiles = None, 
                 buffer = 0.6, bufferdef= None,
                 bandsms = ['blue', 'green', 'red', 'edge', 'nir'],
                 bandsrgb = ["red","green","blue"],
                 shiftmethod = "convolution",
                 xyz_spatialres = 0.012,
                 exportrgb = True,
                 exportxyz = True,
                 rgb_asreference = True):

    """
    Function to merge UAV images from multiple-sources (RGB, Multispectral and 3D-cloud points)

    Parameters:
    ----------
    roi: geopandas
        bounding box to be used for extracting the data

    """
    #roi = allpolygons.loc[id:id,:].copy()
    geom = roi.geometry.values.buffer(buffer, join_style=2)
    # process multispectral files if there is a list
    if msfiles is not None:
        pathms = msfiles[id]
        msm = drone_data.DroneData(pathms, 
                        bands=bandsms, 
                        bounds=geom,
                        table = False)
                
    else:
        msm = None

    # process rgb images if there is a list of files path
    if rgbfiles is not None:
        pathrgb = rgbfiles[id] 
        mmrgb = drone_data.DroneData(pathrgb, 
                        multiband_image=True,
                        bands= bandsrgb,
                        bounds=geom,
                        table=False)
            
    else:
        mmrgb = None
    
    # process xyz files if there is a list of files path
    if xyzfiles is not None:
        pathxyz = xyzfiles[id]
        geombufferxyz = roi.geometry.values.buffer(bufferdef, join_style=2)
        xrdata= fromxyz_file_to_xarray([pathxyz], 
                                      geombufferxyz[0],
                                      sres=xyz_spatialres)

        xrdata = xrdata.isel(date = 0)

        xrdata = xrdata.rename({'red':'red_3d',
                     'blue':'blue_3d',
                     'green':'green_3d'})
    else:
        xrdata = None
        
    imagelist= []

    if msm is not None and mmrgb is not None:
        # shift displacement correction using rgb data
        shiftconv= find_shift_between2xarray(msm.drone_data, mmrgb.drone_data, method=shiftmethod)
        msregistered = register_xarray(msm.drone_data, shiftconv)

        msclipped = clip_xarraydata(msregistered, roi.loc[:,'geometry'], buffer = bufferdef)
        mmrgbclipped = clip_xarraydata(mmrgb.drone_data, roi.loc[:,'geometry'], buffer = bufferdef)

        # stack multiple missions using multispectral images as reference
        if rgb_asreference:
            msclipped = resample_xarray(msclipped, mmrgbclipped, method = 'nearest')

        else:
            mmrgbclipped = resample_xarray(mmrgbclipped,msregistered)
        
        
        msclipped = msclipped.rename({'red':'red_ms',
                     'blue':'blue_ms',
                     'green':'green_ms'})
        imagelist.append(mmrgbclipped)
        imagelist.append(msclipped)

        if xrdata is not None and exportxyz:
            xrdata = resample_xarray(xrdata,mmrgbclipped, method = 'nearest')
            imagelist.append(xrdata)

    elif msm is not None:
        msclipped = clip_xarraydata(msm.drone_data, roi.loc[:,'geometry'], buffer = bufferdef)

        imagelist.append(msclipped)
        if xrdata is not None:
            xrdata = resample_xarray(xrdata,msclipped, method = 'nearest')
            imagelist.append(xrdata)

    elif mmrgb is not None:
        mmrgbclipped = clip_xarraydata(mmrgb.drone_data, roi.loc[:,'geometry'], buffer = bufferdef)
        if exportrgb:
            imagelist.append(mmrgbclipped)
        if xrdata is not None and exportxyz:
            xrdata = resample_xarray(xrdata,mmrgbclipped, method = 'nearest')
            imagelist.append(xrdata)

    else:
        if xrdata is not None and exportxyz:
            imagelist.append(xrdata)

    if len(imagelist)>0:
        if len(imagelist) == 1:
            output = imagelist[0]
        else:
            output = xarray.merge(imagelist)
        
        output.attrs['count'] =len(list(output.keys()))

    else:
        raise ValueError('no mission found')
        
    return output



#### preprocessing functions

def single_vi_bsl_impt_preprocessing(
                        xrpolfile,
                        input_path = None,
                        baseline = True,
                        reference_date = 0,
                        height_name= 'z',
                        bsl_method = 'max_probability',
                        leaf_angle = True,
                        imputation = True,
                        equalization = True,
                        nabandmaskname='red',
                        vilist = None,
                        bsl_value = None
                        ):

    suffix = ''

    if type(xrpolfile) is xarray.core.dataset.Dataset:
       xrdatac = xrpolfile.copy()

    else:
        with open(os.path.join(input_path,xrpolfile),"rb") as f:
            xrdata= pickle.load(f)
        xrdatac = xrdata.copy()
        del xrdata
    if baseline:

        if bsl_value is not None:
            bsl = bsl_value
        else:    
            xrdf = xrdatac.isel(date = reference_date).copy().to_dataframe()
            altref = xrdf.reset_index().loc[:,('x','y',height_name,'red_3d','green_3d','blue_3d')].dropna()
            bsl = get_baseline_altitude(altref, method=bsl_method)

        xrdatac[height_name] = (xrdatac[height_name]- bsl)*100
        xrdatac[height_name] = xrdatac[height_name].where(
            np.logical_or(np.isnan(xrdatac[height_name]),xrdatac[height_name] > 0 ), 0)
        suffix +='bsl_' 


    if imputation:
        if len(list(xrdatac.dims.keys())) >=3:
            xrdatac = impute_4dxarray(xrdatac, 
            bandstofill=[height_name],
            nabandmaskname=nabandmaskname,n_neighbors=5)
        else:
            xrdatac = xarray_imputation(xrdatac, bands=[height_name],nabandmaskname=nabandmaskname,n_neighbors=5)
            
        suffix +='imputation_' 

    if leaf_angle:
        xrdatac = calculate_leaf_angle(xrdatac, invert=True)
        suffix +='la_'
    if equalization:
        xrdatac = hist_ndxarrayequalization(xrdatac, bands = ['red','green','blue'],keep_original=True)
        suffix +='eq_'

    if vilist is not None:
        for vi in vilist:
            xrdatac = calculate_vi_fromxarray(xrdatac,vi = vi,expression = VEGETATION_INDEX[vi])
        suffix +='vi_'

    xrdatac.attrs['count'] = len(list(xrdatac.keys()))
    return xrdatac, suffix



def run_parallel_preprocessing_perpolygon(
    xrpolfile,
    input_path,
    baseline = True,
    reference_date = 0,
    height_name= 'z',
    bsl_method = 'max_probability',

    leaf_angle = True,

    imputation = True,
    nabandmaskname='red',
    vilist = ['ndvi','ndre'],
    output_path = None,
    bsl_value = None
    ):
    
    if output_path is None:
        output_path= ""
    else:
        if not os.path.isdir(output_path):
            os.mkdir(output_path)



    xrdatac,suffix = single_vi_bsl_impt_preprocessing(xrpolfile,
                                    input_path= input_path,
                                    baseline= baseline,
                                    reference_date= reference_date,
                                    height_name = height_name,
                                    bsl_method= bsl_method,
                                    leaf_angle = leaf_angle,
                                    imputation = imputation,
                                    nabandmaskname = nabandmaskname,
                                    vilist = vilist,
                                    bsl_value = bsl_value)


    textafterpol = xrpolfile.split('_pol_')[1]
    idpol = textafterpol.split('_')[0]
    

    if textafterpol.split('_')[1] == 'first' or textafterpol.split('_')[1] == 'last':
        idpol+='_' + textafterpol.split('_')[1]

    outfn = os.path.join(output_path, '{}pol_{}.pickle'.format(suffix,idpol))

    with open(outfn, "wb") as f:
        pickle.dump(xrdatac, f)




import os
from .gis_functions import find_shift_between2xarray,register_xarray,clip_xarraydata,resample_xarray
import xarray
import pickle

import geopandas as gpd
from .drone_data import DroneData
from .xyz_functions import CloudPoints



def stack_multisource_data(roi,ms_data = None, rgb_data = None, pointclouddata = None, bufferdef = None, rgb_asreference = True):

    imagelist = []

    if pointclouddata is not None:
        pointclouddata = pointclouddata.rename({'red':'red_3d',
                     'blue':'blue_3d',
                     'green':'green_3d'})

    if rgb_data is not None and ms_data is not None:
        # shift displacement correction using rgb data
        shiftconv= find_shift_between2xarray(ms_data.drone_data, rgb_data.drone_data)
        msregistered = register_xarray(ms_data.drone_data, shiftconv)

        msclipped = clip_xarraydata(msregistered, roi.loc[:,'geometry'], buffer = bufferdef)
        mmrgbclipped = clip_xarraydata(rgb_data.drone_data, roi.loc[:,'geometry'], buffer = bufferdef)

        # stack multiple missions using multispectral images as reference
        if rgb_asreference:
            msclipped = resample_xarray(msclipped, mmrgbclipped, method = 'nearest')

        else:
            mmrgbclipped = resample_xarray(mmrgbclipped,msregistered)
        
        msclipped = msclipped.rename({'red':'red_ms',
                     'blue':'blue_ms',
                     'green':'green_ms'})
        imagelist.append(mmrgbclipped)
        imagelist.append(msclipped)

        if pointclouddata is not None:
            
            pointclouddatares = resample_xarray(pointclouddata,mmrgbclipped, method = 'nearest')
            imagelist.append(pointclouddatares)

    elif ms_data is not None:
        msclipped = clip_xarraydata(ms_data.drone_data, roi.loc[:,'geometry'], buffer = bufferdef)
        imagelist.append(msclipped)
        if pointclouddata is not None:
            pointclouddatares = resample_xarray(pointclouddata,msclipped, method = 'nearest')
            imagelist.append(pointclouddatares)

    elif rgb_data is not None:
        mmrgbclipped = clip_xarraydata(rgb_data.drone_data, roi.loc[:,'geometry'], buffer = bufferdef)
        imagelist.append(mmrgbclipped)
        if pointclouddata is not None:
            pointclouddatares = resample_xarray(pointclouddata,mmrgbclipped, method = 'nearest')
            imagelist.append(pointclouddatares)

    else:
        if pointclouddata is not None:
            imagelist.append(pointclouddata)

    if len(imagelist)>0:
        if len(imagelist) == 1:
            output = imagelist[0]
        else:
            output = xarray.merge(imagelist)
        output.attrs['count'] =len(list(output.keys()))

    return output

def _set_dronedata(path, **kwargs):
    if os.path.exists(path):
        data = DroneData(path, **kwargs)
    else:
        raise ValueError('the path: {} does not exist'.format(path))
    return data

class IndividualUAVData(object):
    """
    A class to concatenate multiple sourcing data
    """
    @property
    def rgb_data(self):
        if self.uav_sources['rgb'] is not None:
            data = clip_xarraydata(self.uav_sources['rgb'], 
                self.spatial_boundaries.loc[:,'geometry'])
        else:
            data = None
        return data

    @property
    def ms_data(self):
        if self.uav_sources['ms'] is not None:
            data = clip_xarraydata(self.uav_sources['ms'], 
                self.spatial_boundaries.loc[:,'geometry'])
        else:
            data = None
        return data


    def export_as_pickle(self, path = None, uav_image = 'stacked', preffix = None):
        if not os.path.exists(path):
            os.mkdir(path)
        
        if preffix is not None:
            fn = '{}_{}.pickle'.format(preffix, list(self._fnsuffix))
        with open(os.path.join(path, fn), "wb") as f:
                pickle.dump(self.uav_sources[uav_image], f)

    def stack_uav_data(self, bufferdef = None, rgb_asreference = True):

        img_stacked =  stack_multisource_data(self.spatial_boundaries,
                                ms_data = self.uav_sources['ms'], 
                                rgb_data = self.uav_sources['rgb'], 
                                pointclouddata = self.uav_sources['pointcloud'].twod_image, 
                                bufferdef = bufferdef, rgb_asreference = rgb_asreference)

        self.uav_sources.update({'stacked':img_stacked})

        return img_stacked
        
    
    def rgb_uavdata(self, **kwargs):
        rgb_data = _set_dronedata(self.rgb_path, roi = self._boundaries_buffer, multiband_image=True, **kwargs)
        self.uav_sources.update({'rgb':rgb_data})
        self._fnsuffix = self._fnsuffix+ 'rgb'

    def ms_uavdata(self, **kwargs):    
        ms_data = _set_dronedata(self.ms_input, roi = self._boundaries_buffer, multiband_image=False, **kwargs)
        self.uav_sources.update({'ms':ms_data})
        self._fnsuffix = self._fnsuffix+ 'ms'

    def pointcloud(self, interpolate = True, **kwargs):
        
        try:
            if os.path.exists(self.threed_input):
                pcloud_data = CloudPoints(self.threed_input,gpdpolygon= self.spatial_boundaries, verbose = False)
                pcloud_data.to_xarray(interpolate = interpolate,**kwargs)
                self._fnsuffix = self._fnsuffix+ 'pointcloud'
        except:
            pcloud_data = None
            raise Warning('point cloud information was not found, check coordinates')
        #points_perplant.to_xarray(sp_res=0.012, interpolate=False)
        self.uav_sources.update({'pointcloud':pcloud_data})

    def __init__(self, 
                 rgb_input = None,
                 ms_input = None,
                 threed_input = None,
                 spatial_boundaries = None):

        self.uav_sources = {'rgb': None,
                            'ms': None,
                            'pointcloud': None,
                            'stacked': None}
                            
        self._fnsuffix = ''
        self.rgb_path = rgb_input
        self.ms_input = ms_input
        self.threed_input = threed_input
        self.spatial_boundaries = spatial_boundaries
        ### read data with buffer
        self._boundaries_buffer = spatial_boundaries.copy().buffer(0.6, join_style=2)


