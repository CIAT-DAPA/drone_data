from turtle import addshape
import xarray
from datetime import datetime
import os
import pickle
import numpy as np
import geopandas as gpd

from .data_processing import data_standarization

from .data_processing import find_date_instring
from .gis_functions import clip_xarraydata, resample_xarray, register_xarray,find_shift_between2xarray
from .xr_functions import stack_as4dxarray,CustomXarray
from .xyz_functions import CloudPoints
from .xyz_functions import get_baseline_altitude
from .gis_functions import impute_4dxarray,xarray_imputation,hist_ndxarrayequalization
from .xyz_functions import calculate_leaf_angle
from .drone_data import calculate_vi_fromxarray

from .drone_data import DroneData
from .general import MSVEGETATION_INDEX

def run_parallel_mergemissions_perpol(j, bbboxfile, rgb_path = None,
                                      ms_path=None, xyz_path=None,  
                        featurename =None, output_path=None, export =True, 
                        rgb_asreference = True, verbose = False,
                        interpolate = True,
                        resizeinter_method = 'nearest'):


    roiorig = gpd.read_file(bbboxfile)

    capturedates = [find_date_instring(rgb_path[i]) for i in range(len(rgb_path))]
    datesnames = [datetime.strptime(m,'%Y%m%d') for m in capturedates]

    datalist = []

    for i in range(len(rgb_path)):
        
        uavdata = IndividualUAVData(rgb_input = rgb_path[i],
                    ms_input = ms_path[i],
                    threed_input = xyz_path[i],
                    spatial_boundaries = roiorig.iloc[j:j+1])

        uavdata.rgb_uavdata()
        uavdata.ms_uavdata()
        uavdata.pointcloud(interpolate = interpolate)
        uavdata.stack_uav_data(bufferdef = None, rgb_asreference = rgb_asreference)
        datalist.append(uavdata.uav_sources['stacked'])

    alldata = stack_as4dxarray(datalist,axis_name = 'date', 
            valuesaxis_names=datesnames, 
            resizeinter_method = resizeinter_method)

    if export:
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        if featurename is not None:
            preffix = roiorig.iloc[j:j+1].reset_index()[featurename][0]
            fn = '{}_{}_{}.pickle'.format(featurename, preffix, uavdata._fnsuffix)
        else:
            fn = '{}_{}.pickle'.format(preffix, uavdata._fnsuffix)

        if verbose:
            print(j,fn)

        with open(os.path.join(output_path, fn), "wb") as f:
            pickle.dump(alldata, f)

    return alldata



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
                        bandstofill = None,
                        equalization = True,
                        nabandmaskname='red',
                        vilist = None,
                        bsl_value = None,
                        overwritevi = False
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
        if bandstofill is None:
            bandstofill = list(xrdatac.keys())
        if len(list(xrdatac.dims.keys())) >=3:
            xrdatac = impute_4dxarray(xrdatac, 
            bandstofill=bandstofill,
            nabandmaskname=nabandmaskname,n_neighbors=5)
        #else:
        #    xrdatac = xarray_imputation(xrdatac, bands=[height_name],n_neighbors=5)
            
        suffix +='imputation_' 

    if leaf_angle:
        xrdatac = calculate_leaf_angle(xrdatac, invert=True)
        suffix +='la_'
    if equalization:
        xrdatac = hist_ndxarrayequalization(xrdatac, bands = ['red','green','blue'],keep_original=True)
        suffix +='eq_'

    if vilist is not None:
        for vi in vilist:
            xrdatac = calculate_vi_fromxarray(xrdatac,vi = vi,
                                              expression = MSVEGETATION_INDEX[vi], 
                                              overwrite=overwritevi)
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


##########

def stack_multisource_data(roi,ms_data = None, 
                           rgb_data = None, pointclouddata = None, 
                           bufferdef = None, rgb_asreference = True, 
                           resamplemethod = 'nearest'):
    """
    a function to stack multiple UAV source data, curretnly it can merge data from MultiSpectral, RGB, and pointcloud data.
    The point cloud data is extracted from a file type xyz which is pderived product from the RGB camera. This file was obtained from 
    the sfm pix4D analysis. You don;t necesarly must have all three sources of data, you can leave the other sources as None if you don't have them.
    ...
    Parameters
    ----------
    roi: polygon
        region of interest
    ms_data: drone_data class, optional
        multispectral information
    rgb_data: drone_data class, optional
        rgb information this is provided by a RGB high definition camera
    pointclouddata: xarray class, optional
        2d point cloud image
    resample_method: str, optional
        insterpolation method that will be used to resample the image
         ({"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial"}, default: "nearest")

    """
    imagelist = []

    if pointclouddata is not None:
        pointclouddata = pointclouddata.rename({'red':'red_3d',
                     'blue':'blue_3d',
                     'green':'green_3d'})

    if rgb_data is not None and ms_data is not None:
        # shift displacement correction using rgb data
        shiftconv= find_shift_between2xarray(ms_data, rgb_data)
        msregistered = register_xarray(ms_data, shiftconv)

        # stack multiple missions using multispectral images as reference
        if rgb_asreference:
            msregistered = resample_xarray(msregistered, rgb_data, method = resamplemethod)

        else:
            rgb_data = resample_xarray(rgb_data,msregistered)
        
        ## clip to the original boundaries
        msclipped = clip_xarraydata(msregistered, roi.loc[:,'geometry'], buffer = bufferdef)
        mmrgbclipped = clip_xarraydata(rgb_data, roi.loc[:,'geometry'], buffer = bufferdef)

        msclipped = msclipped.rename({'red':'red_ms',
                     'blue':'blue_ms',
                     'green':'green_ms'})
        imagelist.append(mmrgbclipped)
        imagelist.append(msclipped)

        if pointclouddata is not None:
            pointclouddatares = resample_xarray(pointclouddata,mmrgbclipped, method = resamplemethod)
            imagelist.append(pointclouddatares)

    elif ms_data is not None:
        msclipped = clip_xarraydata(ms_data, roi.loc[:,'geometry'], buffer = bufferdef)
        imagelist.append(msclipped)
        if pointclouddata is not None:
            pointclouddatares = resample_xarray(pointclouddata,msclipped, method = resamplemethod)
            imagelist.append(pointclouddatares)

    elif rgb_data is not None:
        mmrgbclipped = clip_xarraydata(rgb_data, roi.loc[:,'geometry'], buffer = bufferdef)
        imagelist.append(mmrgbclipped)
        if pointclouddata is not None:
            pointclouddatares = resample_xarray(pointclouddata,mmrgbclipped, method = resamplemethod)
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

RGB_BANDS = ["red","green","blue"] ### this order is because the rgb uav data is stacked
MS_BANDS = ['blue', 'green', 'red', 'edge', 'nir']

class IndividualUAVData(object):
    """
    A class to concatenate multiple uav orthomosiac sourcing data
    using a spatial vector file
    """
    

    @property
    def rgb_data(self):
        if self.uav_sources['rgb']:
            data = clip_xarraydata(self.uav_sources['rgb'].drone_data, 
                self.spatial_boundaries.loc[:,'geometry'])
        else:
            data = None
        return data

    @property
    def ms_data(self):
        if self.uav_sources['ms']:
            data = clip_xarraydata(self.uav_sources['ms'].drone_data, 
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

    def stack_uav_data(self, bufferdef = None, rgb_asreference = True, resample_method = 'nearest'):

        pointcloud = self.uav_sources['pointcloud'].twod_image if self.uav_sources['pointcloud'] is not None else None
        ms = self.uav_sources['ms'].drone_data if self.uav_sources['ms'] is not None else None
        rgb = self.uav_sources['rgb'].drone_data if self.uav_sources['rgb'] is not None else None

        img_stacked =  stack_multisource_data(self.spatial_boundaries,
                                ms_data = ms, 
                                rgb_data = rgb, 
                                pointclouddata = pointcloud, 
                                bufferdef = bufferdef, rgb_asreference = rgb_asreference,
                                resamplemethod = resample_method)

        self.uav_sources.update({'stacked':img_stacked})

        return img_stacked
        
    
    def rgb_uavdata(self, **kwargs):
        """
        read rgb data
        """
        
        if self.rgb_bands is None:
            self.rgb_bands  = RGB_BANDS
        
        rgb_data = _set_dronedata(
            self.rgb_path, 
            bounds = self._boundaries_buffer.copy(), 
            multiband_image=True, 
            bands = self.rgb_bands, **kwargs)
        self.uav_sources.update({'rgb':rgb_data})
        self._fnsuffix = self._fnsuffix+ 'rgb'

    def ms_uavdata(self, **kwargs):    
        """
        read multi spectral data
        """
        if self.ms_bands is None:
            self.ms_bands = MS_BANDS


        ms_data = _set_dronedata(self.ms_input, 
                    bounds = self._boundaries_buffer.copy(),
                    multiband_image=False, bands = self.ms_bands, **kwargs)

        self.uav_sources.update({'ms':ms_data})
        self._fnsuffix = self._fnsuffix+ 'ms'

    def pointcloud(self, interpolate = True, **kwargs):
        
        try:
            if os.path.exists(self.threed_input):
                buffertmp = self._boundaries_buffer.copy().reset_index()
                buffertmp = buffertmp[0] if type(buffertmp) == tuple else buffertmp
                if 0 in buffertmp.columns:
                    buffertmp = buffertmp.rename(columns={0:'geometry'})
                
                pcloud_data = CloudPoints(self.threed_input,
                                #gpdpolygon= self.spatial_boundaries.copy(), 
                                gpdpolygon=buffertmp,
                                verbose = False)
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
                 spatial_boundaries = None,
                 rgb_bands = None,
                 ms_bands = None,
                 buffer = 0.6,
        ):

        self.rgb_bands = rgb_bands
        self.ms_bands = ms_bands

        self.uav_sources = {'rgb': None,
                            'ms': None,
                            'pointcloud': None,
                            'stacked': None}
                            
        self._fnsuffix = ''
        self.rgb_path = rgb_input
        self.ms_input = ms_input
        self.threed_input = threed_input
        self.spatial_boundaries = spatial_boundaries.copy()
        ### read data with buffer
        self._boundaries_buffer = spatial_boundaries.copy().buffer(buffer, join_style=2)


##
import tqdm
import random
import itertools

def extract_random_samples_from_drone(imagepath, geometries, bands = None, multiband = True, buffer = 0, samples = 500):
    """extract sample polygons data from UAV

    Args:
        imagepath (_type_): _description_
        geometries (_type_): _description_
        bands (_type_, optional): _description_. Defaults to None.
        multiband (bool, optional): _description_. Defaults to True.
        buffer (int, optional): _description_. Defaults to 0.
        samples (int, optional): _description_. Defaults to 500.

    Returns:
        _type_: _description_
    """
    dictdata = {}
    first = True
    for j in tqdm.tqdm(random.sample(range(geometries.shape[0]), samples)):
        try:
            
            drdata = DroneData(imagepath,    multiband_image= multiband, bounds= geometries.iloc[j:j+1].buffer(buffer, join_style=2),bands = bands)

            xrdata = drdata.drone_data.copy()

            for i, feature in enumerate(bands):
                data = xrdata[feature].values
                data[data == 0] = np.nan
                if first:
                    valtosave = list(itertools.chain.from_iterable(data))
                else:
                    valtosave = dictdata[feature] + list(itertools.chain.from_iterable(data))
                
                dictdata[feature] = valtosave
            first = False
        except:
            pass
    
    return dictdata


def summary_data(data):
    summary = {}
    alldata = []
    for feature in list(data.keys()):
        datafe = data[feature]
        meanstd = [np.nanmean(datafe),
                    np.nanstd(datafe)]
        minmax = [np.nanmin(datafe),
                    np.nanmax(datafe)]
        alldata.append(datafe)
        summary[feature] = {'normalization':minmax,
                            'standarization':meanstd}

    alldata = np.array(list(itertools.chain.from_iterable(alldata)))
    minmax = [np.nanmin(alldata),
            np.nanmax(alldata)]
    meanstd = [np.nanmean(alldata),
            np.nanstd(alldata)]
    
    summary.update({'normalization':minmax,
                    'standarization':meanstd})

    return summary



##

def extract_uav_datausing_geometry(rgbpath = None,mspath =None,pcpath = None,geometry = None,rgb_channels = None,
                                   mschannels = None, 
                                   buffer = 0,
                                   processing_buffer = 0, 
                                   interpolate_pc = True, 
                                   rgb_asreference = True):
    
    uavdata = IndividualUAVData(rgbpath, 
                    mspath, 
                    pcpath, 
                    geometry, 
                    rgb_channels, 
                    mschannels, buffer = processing_buffer)
    
    if rgbpath is not None:
        uavdata.rgb_uavdata()
    if mspath is not None:
        uavdata.ms_uavdata()
    if pcpath is not None:
        uavdata.pointcloud(interpolate = interpolate_pc)
        
    xrinfo = uavdata.stack_uav_data(bufferdef = buffer, 
        rgb_asreference = rgb_asreference,resample_method = 'nearest')
    
    return xrinfo

def xr_data_normalization(xrdata, scaler, scalertype = 'standarization'):
    
    if scalertype == 'standarization':
        fun = data_standarization
    
    varchanels = list(xrdata.keys())
    
    for channel in varchanels:
        if channel in list(scaler.keys()):
            val1, val2 = scaler[channel]
            scaleddata = fun(xrdata[channel].to_numpy(), val1, val2)
            xrdata[channel].values = scaleddata
    return xrdata

class MultiMLTImages(CustomXarray):
    @property
    def listcxfiles(self):
        if self.path is not None:
            assert os.path.exists(self.path) ## directory soes nmot exist
            files = [i for i in os.listdir(self.path) if i.endswith('pickle')]
        else:
            files = None
        return files
    
    def extract_samples(self, channels = None, n_samples = 100, **kwargs):
        import random
        import itertools
        import tqdm
        
        dictdata = {}
        
        listids= []
        while len(listids)< n_samples:
            tmplistids = list(np.unique(random.sample(range(self.geometries.shape[0]), n_samples-len(listids))))
            listids = list(np.unique(listids+tmplistids))
        
        for j in tqdm.tqdm(listids[:n_samples]):
            try:
                self.individual_data(j, **kwargs)
                channels = list(self.xrdata.keys()) if channels is None else channels
                #
                for i, feature in enumerate(channels):
                    data = self.xrdata[feature].values.copy()
                    data[data == 0] = np.nan
                    if feature not in list(dictdata.keys()):
                        valtosave = list(itertools.chain.from_iterable(data))
                    else:
                        valtosave = dictdata[feature] + list(itertools.chain.from_iterable(data))
                    
                    dictdata[feature] = valtosave
            except:
                pass
        return dictdata
    
    def individual_data(self,  geometry_id, interpolate_pc = True,
                        rgb_asreference = True, 
                        datesnames = None,
                        buffer = None):
        
        
        assert self.geometries.shape[0] > geometry_id
        roi = self.geometries.iloc[geometry_id:geometry_id+1]

        datalist = []
        for i in range(len(self.rgb_paths)):
            rgbpath = self.rgb_paths[i] if self.rgb_paths is not None else None
            mspath = self.ms_paths[i] if self.ms_paths is not None else None
            pcpath = self.pointcloud_paths[i] if self.pointcloud_paths is not None else None
            
            datalist.append(extract_uav_datausing_geometry(rgbpath, 
                            mspath, 
                            pcpath, 
                            roi, 
                            self.rgb_channels, 
                            self.ms_channels, 
                            buffer= buffer,
                            processing_buffer = self.processing_buffer,
                            interpolate_pc = interpolate_pc, rgb_asreference = rgb_asreference))
        
        if len(datalist)>1:
            self.xrdata = stack_as4dxarray(datalist,axis_name = 'date', 
                valuesaxis_names=datesnames, 
                resizeinter_method = 'nearest')
        else:
            self.xrdata = datalist[0]
    
        return self.xrdata
    
    def read_individual_data(self, pattern = None, onlythesechannels = None):
        """read 
        """
        file = [i for i in self.listcxfiles if i == pattern][0]
        customdict = self._read_data(path=self.path, 
                                   fn = os.path.basename(file),
                                   suffix='pickle')
        
        return self.to_array(customdict,onlythesechannels)
        
    def scale_uavdata(self, scaler, scaler_type = 'standarization'):
        """scale the imagery using values

        Args:
            scaler (_type_): _description_
            scaler_type (dict, optional): _description_. Defaults to 'standarization'.
        """
        assert type(scaler) == dict
        #assert len(list(scaler.keys())) == len(list(self.xrdata.keys()))
        
        self.xrdata = xr_data_normalization(self.xrdata, scaler, scalertype = scaler_type)
        
    def calculate_vi(self, vilist, viequations = None, overwritevi = True):
        if vilist is None:
            vilist = ['ndvi']
        if viequations is None:
            from drone_data.utils.general import MSVEGETATION_INDEX
            viequations = MSVEGETATION_INDEX
        xrdatac = self.xrdata.copy()
        for vi in vilist:
            xrdatac = calculate_vi_fromxarray(xrdatac,vi = vi,
                                              expression = viequations[vi], 
                                              overwrite=overwritevi)
        
        self.xrdata = xrdatac
        return self.xrdata
    
    def summarize_as_dataframe(self, channels = None,func = np.nanmedian):
        dataasdict = None
        channels = channels if channels is not None else list(self.xrdata.keys())
        if len(list(self.xrdata.dims.keys())) == 2:
            dataasdict = {channel: [func(self.xrdata[channel].values)] 
             for channel in channels}
        
        return dataasdict
    
    def __init__(self, 
                 rgb_paths=None, 
                 ms_paths=None, 
                 pointcloud_paths=None, 
                 spatial_file=None, 
                 rgb_channels=None, 
                 ms_channels=None, 
                 path = None,
                 processing_buffer=0.6,
                 **kwargs):
        """function to extract orthomosaic imagery from RGB, MS, and point cloud using 
        a vector file

        Args:
            rgb_paths (list, optional): list of folder paths that contains the RGB orthomosaic. Defaults to None.
            ms_paths (list, optional): list of folder paths that contains the MS orthomosaic imagery. Defaults to None.
            pointcloud_paths (list, optional): list of folder paths that contains the point cloud with extension xyz. Defaults to None.
            spatial_file (str, optional): vector file path. Defaults to None.
            rgb_channels (list, optional): rgb channel names. Defaults to None.
            ms_channels (list, optional): multi-spectral channel names. Defaults to None.
            path (str, optional): a path that contains customdict xarray as pickle. Default to None.
            processing_buffer (float, optional): _description_. Defaults to 0.6.
        """
        
        self.rgb_paths = rgb_paths
        self.ms_paths = ms_paths
        self.pointcloud_paths = pointcloud_paths
        self.processing_buffer = processing_buffer
        self.rgb_channels = rgb_channels
        self.ms_channels = ms_channels
        self.path = path
        super().__init__(**kwargs)
        if spatial_file is not None:
            assert os.path.exists(spatial_file)
            self.geometries = gpd.read_file(spatial_file)
        
            