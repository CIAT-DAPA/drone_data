from turtle import addshape
import xarray
from datetime import datetime
import os
import pickle
import numpy as np
import geopandas as gpd


import multiprocessing

import concurrent.futures as cf
from .data_processing import find_date_instring
from .gis_functions import clip_xarraydata, resample_xarray, register_xarray,find_shift_between2xarray
from .xr_functions import stack_as4dxarray,CustomXarray, from_dict_toxarray,from_xarray_to_dict
from .xyz_functions import CloudPoints
from .xyz_functions import get_baseline_altitude
from .gis_functions import impute_4dxarray,xarray_imputation,hist_ndxarrayequalization
from .xyz_functions import calculate_leaf_angle
from .drone_data import calculate_vi_fromxarray

from .drone_data import DroneData
from .general import MSVEGETATION_INDEX

from typing import List, Optional, Union, Any

import tqdm
import random
import itertools

##
RGB_BANDS = ["red","green","blue"] ### this order is because the rgb uav data is stacked
MS_BANDS = ['blue', 'green', 'red', 'edge', 'nir']

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

def stack_multisource_data(roi,ms_data: Optional[DroneData] = None, 
                           rgb_data: Optional[DroneData] = None, 
                           pointclouddata: Optional[xarray.DataArray] = None, 
                           bufferdef: Optional[float] = None, 
                           rgb_asreference: bool = True, 
                           resamplemethod: str = 'nearest'):
    
    """
    Stacks multiple UAV source data (Multispectral, RGB, Point Cloud) into a single dataset.

    Parameters:
    ----------
    roi : Polygon or similar
        Region of interest for clipping the data.
    ms_data : DroneData, optional
        Multispectral data.
    rgb_data : DroneData, optional
        RGB data from a high-definition camera.
    pointclouddata : xarray.DataArray, optional
        2D point cloud image data.
    bufferdef : float, optional
        Buffer definition for clipping. (Explain its usage more clearly)
    rgb_asreference : bool, optional
        Use RGB data as the reference for alignment if True.
    resamplemethod : str, optional
        Method for resampling ({"linear", "nearest", "zero", "slinear", "quadratic", "cubic", "polynomial"}, default: "nearest").

    Returns:
    -------
    xarray.DataArray or None
        Stacked UAV data as an xarray DataArray or None if no data is provided.
        
    Notes:
    -------
    The point cloud data is extracted from a file type xyz which is pderived product from the RGB camera. This file was obtained from 
    the sfm pix4D analysis. You don;t necesarly must have all three sources of data, you can leave the other sources as None if you don't have them.
    """
    
    imagelist = []

    if pointclouddata is not None:
        pointclouddata = pointclouddata.rename({'red':'red_3d',
                     'blue':'blue_3d',
                     'green':'green_3d'})

    # Processing RGB and MS data
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

        # Rename MS channels to avoid naming conflicts
        msclipped = msclipped.rename({'red':'red_ms',
                     'blue':'blue_ms',
                     'green':'green_ms'})
        imagelist.append(mmrgbclipped)
        imagelist.append(msclipped)

        # Resample point cloud data if available
        if pointclouddata is not None:
            pointclouddatares = resample_xarray(pointclouddata,mmrgbclipped, method = resamplemethod)
            imagelist.append(pointclouddatares)

    # Handling cases with only MS or RGB data
    elif ms_data is not None or rgb_data is not None:
        source = ms_data if ms_data is not None else rgb_data
        souce_c = clip_xarraydata(source, roi.loc[:,'geometry'], buffer = bufferdef)
        imagelist.append(souce_c)
        if pointclouddata is not None:
            pointclouddatares = resample_xarray(pointclouddata,souce_c, method = resamplemethod)
            imagelist.append(pointclouddatares)

    elif pointclouddata is not None:
            imagelist.append(pointclouddata)

    if len(imagelist)>0:
        output = imagelist[0] if len(imagelist) == 1 else xarray.merge(imagelist)
        output.attrs['count'] =len(list(output.keys()))
    else:
        output = None
        
    return output

def _set_dronedata(path, **kwargs):
    if os.path.exists(path):
        data = DroneData(path, **kwargs)
    else:
        raise ValueError('the path: {} does not exist'.format(path))
    return data



class IndividualUAVData(object):
    """
    A class to concatenate multiple uav orthomosiac sourcing data
    using a spatial vector file
    """
    
    def __init__(self, 
                 rgb_input: Optional[List[str]] = None,
                 ms_input: Optional[List[str]] = None,
                 threed_input: Optional[List[str]] = None,
                 spatial_boundaries = None,
                 rgb_bands = None,
                 ms_bands = None,
                 buffer = 0.6,
        ):
        
        """
        Initializes the IndividualUAVData class for processing UAV data.

        Parameters:
        ----------
        rgb_input : List(str), optional
            Path containing the RGB orthomosaic imagery.
        ms_input : List(str), optional
            Path containing the MS orthomosaic imagery.
        threed_input : List(str), optional
            Path containing the point cloud data (XYZ format).
        spatial_boundaries : GeoDataFrame or similar, optional
            Spatial polygon defining the area of interest.
        rgb_bands : list, optional
            Names of RGB channels.
        ms_bands : list, optional
            Names of multispectral channels.
        buffer : float, optional
            Buffer value for image processing during stacking.

        Notes:
        -----
        The class supports handling and stacking of RGB, multispectral, and point cloud data.
        It provides functionalities for data clipping based on spatial boundaries and exporting
        the processed data.
        """
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
    
    @property
    def rgb_data(self):
        """
        Retrieves RGB processed data.

        Returns:
        -------
        xarray.DataArray or None:
            Clipped RGB data as an xarray DataArray. Returns None if RGB data is not available.
        """
    
        if self.uav_sources['rgb']:
            data = clip_xarraydata(self.uav_sources['rgb'].drone_data, 
                self.spatial_boundaries.loc[:,'geometry'])
        else:
            data = None
        return data

    @property
    def ms_data(self):
        """
        Retrieves clipped multispectral (MS) data using the defined spatial boundaries.

        Returns:
        -------
        xarray.DataArray or None:
            Clipped MS data as an xarray DataArray. Returns None if MS data is not available.
        """
    
        if self.uav_sources['ms']:
            data = clip_xarraydata(self.uav_sources['ms'].drone_data, 
                self.spatial_boundaries.loc[:,'geometry'])
        else:
            data = None
        return data


    def export_as_pickle(self, path = None, uav_image = 'stacked', preffix = None):
        
        """
        Exports specified UAV data as a pickle file.

        Parameters:
        ----------
        path : str
            Path to the export directory.
        uav_image : str
            Key for the UAV data to be exported (e.g., 'stacked').
        preffix : str, optional
            Prefix for the output filename.

        Returns:
        -------
        None
        """
    
        if not os.path.exists(path):
            os.mkdir(path)
        
        if preffix is not None:
            fn = '{}_{}.pickle'.format(preffix, list(self._fnsuffix))
        with open(os.path.join(path, fn), "wb") as f:
                pickle.dump(self.uav_sources[uav_image], f)

    def stack_uav_data(self, bufferdef: Optional[float] = None, rgb_asreference: bool = True, resample_method: str = 'nearest'):
        """
        Stacks available UAV data sources (RGB, MS, point cloud) into a single dataset.

        Parameters:
        ----------
        bufferdef : float, optional
            Buffer value for image processing.
        rgb_asreference : bool, optional
            Whether to use RGB data as a reference for stacking.
        resample_method : str, optional
            Resampling method to use (e.g., 'nearest').

        Returns:
        -------
        xarray.DataArray:
            Stacked UAV data as an xarray DataArray.
        """

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
        Processes and stores RGB UAV data.

        Additional keyword arguments are passed to the data processing function.

        Returns:
        -------
        None
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
        Processes and stores multispectral (MS) UAV data.

        Additional keyword arguments are passed to the data processing function.

        Returns:
        -------
        None
        """
        if self.ms_bands is None:
            self.ms_bands = MS_BANDS


        ms_data = _set_dronedata(self.ms_input, 
                    bounds = self._boundaries_buffer.copy(),
                    multiband_image=False, bands = self.ms_bands, **kwargs)

        self.uav_sources.update({'ms':ms_data})
        self._fnsuffix = self._fnsuffix+ 'ms'

    def pointcloud(self, interpolate = True, **kwargs):
        """
        Processes and stores point cloud data.

        Parameters:
        ----------
        interpolate : bool, optional
            Whether to interpolate point cloud data.

        Raises:
        ------
        Warning:
            If point cloud data is not found or coordinates are incorrect.

        Returns:
        -------
        None
        """
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




##


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
                                    
def extract_uav_datausing_geometry(rgbpath: Optional[str] = None,
                                   mspath: Optional[str] = None,
                                   pcpath: Optional[str] = None,
                                   geometry: Optional[Any] = None,
                                   rgb_channels: Optional[List[str]] = None,
                                   mschannels: Optional[List[str]] = None, 
                                   buffer: int = 0,
                                   processing_buffer: float = 0, 
                                   interpolate_pc: bool = True, 
                                   rgb_asreference: bool = True):
    
    """
    Extracts UAV data using provided paths and geometry information.

    Args:
        rgbpath (str, optional): Path to RGB data. Defaults to None.
        mspath (str, optional): Path to multispectral data. Defaults to None.
        pcpath (str, optional): Path to point cloud data. Defaults to None.
        geometry (Any, optional): Geometry information. Defaults to None.
        rgb_channels (List[str], optional): List of RGB channels. Defaults to None.
        mschannels (List[str], optional): List of multispectral channels. Defaults to None.
        buffer (float, optional): Buffer size. Defaults to 0.
        processing_buffer (int, optional): Processing buffer size. Defaults to 0.
        interpolate_pc (bool, optional): Flag to interpolate point cloud data. Defaults to True.
        rgb_as_reference (bool, optional): Flag to use RGB as reference. Defaults to True.

    Returns:
        Any: Information extracted from UAV data.
    """
    
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




class MultiMLTImages(CustomXarray):
    """
    A class for extracting, processing, and stacking multi-temporal remote sensing data.

    Attributes:
        rgb_paths (list): Paths to RGB orthomosaic imagery.
        ms_paths (list): Paths to multispectral orthomosaic imagery.
        pointcloud_paths (list): Paths to point cloud data.
        processing_buffer (float): A buffer value for image processing.
        rgb_channels (list): Names of RGB channels.
        ms_channels (list): Names of multispectral channels.
        path (str): Directory path for customdict xarray files.
        geometries (GeoDataFrame): Geopandas DataFrame of spatial geometries.
    """
    
    def __init__(self, 
                rgb_paths: Optional[List[str]] = None, 
                ms_paths: Optional[List[str]] = None, 
                pointcloud_paths: Optional[List[str]] = None, 
                spatial_file: str=None, 
                rgb_channels: Optional[List[str]]=None, 
                ms_channels: Optional[List[str]]=None, 
                path: str = None,
                processing_buffer: float=0.6,
                **kwargs):
        
        """
        Initializes the MultiMLTImages class for processing multi-temporal remote sensing data.

        This class is designed to handle and process multi-temporal remote sensing data 
        from various sources such as RGB, multispectral (MS), and point cloud data. It 
        supports operations like data extraction, scaling, and calculation of vegetation indices.

        Args:
            rgb_paths (list, optional): List of paths to RGB orthomosaic imagery directories.
            ms_paths (list, optional): List of paths to multispectral orthomosaic imagery directories.
            pointcloud_paths (list, optional): List of paths to point cloud data directories (XYZ format).
            spatial_file (str, optional): Path to a vector file (e.g., GeoJSON, Shapefile) defining spatial geometries for extraction.
            rgb_channels (list, optional): Names of the channels in the RGB data (e.g., ['R', 'G', 'B']).
            ms_channels (list, optional): Names of the channels in the multispectral data.
            path (str, optional): Path to a directory where customdict xarray files are stored as pickle files.
            processing_buffer (float, optional): Buffer value (in meters) used during image processing to handle edge effects.
            **kwargs: Additional keyword arguments to be passed to the parent class (CustomXarray).

        Raises:
            FileNotFoundError: If the provided `spatial_file` does not exist.

        Examples:
            #### Initializing with RGB and MS data paths
            multi_mlt_images = MultiMLTImages(rgb_paths=["/path/to/rgb"], 
                                              ms_paths=["/path/to/ms"],
                                              spatial_file="/path/to/spatial/file.sh")
        """
        
        self.rgb_paths = rgb_paths
        self.ms_paths = ms_paths
        self.pointcloud_paths = pointcloud_paths
        self.processing_buffer = processing_buffer
        self.rgb_channels = rgb_channels
        self.ms_channels = ms_channels
        self.path = path
        super().__init__(**kwargs)
        if spatial_file is not None and not os.path.exists(spatial_file):
            raise FileNotFoundError(f"Spatial file not found: {spatial_file}")

        self.geometries = gpd.read_file(spatial_file) if spatial_file is not None else None
            
    
    @property
    def listcxfiles(self):
        """
        Retrieves a list of filenames ending with 'pickle' in the specified path.

        Returns:
            List[str]: List of filenames.
        """
        
        if self.path is not None:
            assert os.path.exists(self.path) ## directory soes nmot exist
            files = [i for i in os.listdir(self.path) if i.endswith('pickle')]
        else:
            files = None
        return files
    
    def _export_individual_data(self, geometry_id,path,fn,**kwargs):
        """
        Exports data for an individual geometry to a pickle file.

        Args:
            geometry_id (int): Index of the geometry.
            path (str): Path to export directory.
            fn (str): Filename for export.
        """
        
        self.individual_data(geometry_id, **kwargs)
        self.export_as_dict(path = path,fn=fn)
        
    def extract_samples(self, channels: List[str] = None, n_samples: int = 100, **kwargs):
        """
        Extracts sample data from geopandas polygons.

        Args:
            channels (List[str], optional): List of channels to extract.
            n_samples (int): Number of samples to extract.

        Returns:
            Dict[str, List[float]]: Extracted data organized by channel.
        """
        
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
       
    def export_multiple_data(self, path, fnincolumn = None,njobs = 1,verbose=False, **kwargs):
        """
        Exports data for multiple geometries to specified path.

        Args:
            path (str): Path to export directory.
            fnincolumn (str, optional): Column name in geometry DataFrame containing filenames.
            njobs (int, optional): Number of parallel jobs for export.
            verbose (bool, optional): Whether to display a progress bar.

        Returns:
            None
        
        Raises:
            Exception: If any of the parallel tasks fail.
        """
        
        # defining export file names
        geometries_range = range(self.geometries.shape[0])
        
        if fnincolumn is not None:
            fns = ['{}_{}'.format(fnincolumn, i) for i in self.geometries[fnincolumn].values]
        else:
            fns = ['file_{}'.format(i) for i in list(geometries_range)]
            
        if njobs == 1:
            geometries_loop= tqdm.tqdm(geometries_range) if verbose else geometries_range
            for j in geometries_loop:
                self._export_individual_data(self, j,path,fns[j],**kwargs)
                
        else:
            with cf.ProcessPoolExecutor(max_workers=njobs) as executor:
            
                futures = {executor.submit(self._export_individual_data, 
                                           j, path, fns[j], **kwargs) for j in geometries_range}
                
                for future in cf.as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"A task generated an exception: {exc}")
                        raise
        
    def _time_pointsextraction(self,tp:int):
        """
        Extracts data for a specific time point.

        Args:
            tp (int): Time point index.

        Returns:
            DataArray: Data extracted at the specified time point.
        """
        
        rgbpath = self.rgb_paths[tp] if self.rgb_paths is not None else None
        mspath = self.ms_paths[tp] if self.ms_paths is not None else None
        pcpath = self.pointcloud_paths[tp] if self.pointcloud_paths is not None else None
        
        return extract_uav_datausing_geometry(rgbpath, 
                        mspath, 
                        pcpath, 
                        self._tmproi, 
                        self.rgb_channels, 
                        self.ms_channels, 
                        buffer= self._buffer,
                        processing_buffer = self.processing_buffer,
                        interpolate_pc = self._interpolate_pc, rgb_asreference = self._rgb_asreference)
        

    def individual_data(self,  geometry_id, interpolate_pc = True,
                        rgb_asreference = True, 
                        datesnames = None,
                        buffer = None,
                        paralleltimepoints = False, njobs = None):
        """
        Extracts data for an individual geometry.

        Args:
            geometry_id (int): Index of the geometry.
            interpolate_pc (bool, optional): Interpolate point cloud data with knn. Defaults to True.
            rgb_asreference (bool, optional): Use RGB as a reference for co-registration. Defaults to True.
            datesnames (list, optional): Names for dates axis in the data cube. Defaults to None.
            buffer (float, optional): Buffer value for geometry extraction. Defaults to None.
            paralleltimepoints (bool, optional): Whether to process time points in parallel. Defaults to False.
            njobs (int, optional): Number of parallel jobs. Defaults to None.

        Returns:
            DataArray: Extracted data for the individual geometry.
        """
        
        self._scalarflag = False
        assert self.geometries.shape[0] > geometry_id
        roi = self.geometries.iloc[geometry_id:geometry_id+1]

        datalist = []
        self._tmproi,self._buffer,self._rgb_asreference,self._interpolate_pc =  roi, buffer,rgb_asreference,interpolate_pc
        
        if paralleltimepoints:
            if njobs is None:
                njobs = multiprocessing.cpu_count()
            
            results = []
            with cf.ProcessPoolExecutor(max_workers=njobs) as executor:
                for i in range(len(self.rgb_paths)):
                    results.append(executor.submit(self._time_pointsextraction, 
                                                    i))
            #print(results)
            datalist = [future.result() for future in results]
                    
        else:
            for i in range(len(self.rgb_paths)):
                datalist.append(self._time_pointsextraction(i))
        
        if len(datalist)>1:
            if datesnames is None:
                capturedates = [find_date_instring(self.rgb_paths[i]) for i in range(len(self.rgb_paths))]
                datesnames = [datetime.strptime(m,'%Y%m%d') for m in capturedates]
            
            self.xrdata = stack_as4dxarray(datalist,axis_name = 'date', 
                valuesaxis_names=datesnames, 
                resizeinter_method = 'nearest')
        else:
            self.xrdata = datalist[0]
    
        return self.xrdata
    
    def read_individual_data(self, file: str = None, path: str = None, dataformat: str = 'CHW'):
        """
        Reads individual data from a pickle file.

        Args:
            file (str, optional): Filename to read.
            path (str, optional): Directory path containing the file.
            dataformat (str, optional): Format of the data

        Returns:
        xarray.DataArray
            Data read from the file as xarray.
        """
        
        if path is not None:
            file = os.path.basename(file)

        else:
            path = self.path
            file = [i for i in self.listcxfiles if i == file][0]            
        self._scalarflag = False
        customdict = self._read_data(path=path, 
                                   fn = file,
                                   suffix='pickle')
        self.xrdata  = from_dict_toxarray(customdict, dimsformat = dataformat)
        #return self.to_array(self.customdict,onlythesechannels)
    
    #@staticmethod
    def scale_uavdata(self, scaler, scaler_type = 'standarization', applyagain =False):
        from .xr_functions import xr_data_transformation
        """
        Scales multitemporal data using the specified scaler.

        Parameters:
        ----------
        scaler : Dict[str, Scaler]
            Scaler for each channel.
        scaler_type : str, optional
            Type of scaler ('standarization' or 'normalization').
        applyagain : bool, optional
            Whether to apply scaling again if already applied.

        Returns:
        -------
        xarray.DataArray
            Scaled UAV data.
        """
        
        assert type(scaler) == dict
        if not self._scalarflag:
            #assert len(list(scaler.keys())) == len(list(self.xrdata.keys()))
            self.xrdata = xr_data_transformation(self.xrdata, scaler, scalertype = scaler_type)
            self._scalarflag = True
        elif applyagain:
            self.xrdata = xr_data_transformation(self.xrdata, scaler, scalertype = scaler_type)
        else:
            print('this was was already applied, to apply again change to True')
        
        return self.xrdata 
            
                    
    def calculate_vi(self, vilist, viequations: List[str] = None, overwritevi: bool = True, verbose: bool = False):
        """
        Calculates specified vegetation indices on the data.

        Parameters:
        ----------
        vilist : List[str]
            List of vegetation indices to calculate.
        viequations : Dict[str, str], optional
            Custom equations
            
        overwritevi: bool, optional 
            Overwrite existing index in the datacube. Defaults to True.
            
        """
        
        if vilist is None:
            vilist = ['ndvi']
        if viequations is None:
            from drone_data.utils.general import MSVEGETATION_INDEX
            viequations = MSVEGETATION_INDEX
        xrdatac = self.xrdata.copy()
        for vi in vilist:
            if verbose:
                print('Computing {}'.format(vi))
            xrdatac = calculate_vi_fromxarray(xrdatac,vi = vi,
                                              expression = viequations[vi], 
                                              overwrite=overwritevi)
        
        self.xrdata = xrdatac
        return self.xrdata
    
    def summarize_as_dataframe(self, channels = None,func = np.nanmedian):
        """
        Summarizes the datacube as a DataFrame.

        Args:
            channels (list, optional): List of channels to summarize. Defaults to None.
            func (function, optional): Aggregation function. Defaults to np.nanmedian.

        Returns:
            dict: Summary data organized by channel.
        """
        
        dataasdict = None
        channels = channels if channels is not None else list(self.xrdata.keys())
        if len(list(self.xrdata.dims.keys())) == 2:
            dataasdict = {channel: [func(self.xrdata[channel].values)] 
             for channel in channels}
        
        return dataasdict
    

        
            