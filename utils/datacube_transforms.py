from abc import ABCMeta, abstractmethod
import geopandas as gpd

import numpy as np
import os
import xarray

from .xr_functions import (
    CustomXarray, 
    XRColorSpace,
    customdict_transformation,
    transform_listarrays,
    get_data_from_dict,
    from_dict_toxarray, 
    from_xarray_to_dict,
    xr_data_transformation)

from .gis_functions import (
    get_minmax_pol_coords, 
    get_xarray_polygon,
    estimate_pol_buffer, 
    clip_xarraydata)

from .image_functions import resize_2dimg, fill_na_values
from .drone_data import calculate_vi_fromxarray
from typing import List, Optional, Dict



class DataCubeMetrics(metaclass=ABCMeta):
    """
    An abstract base class for handling data cubes, extending functionality for color space calculations,
    vegetation indices, and scaling of data.

    Attributes
    ----------
    xrdata : xr.Dataset
        The xarray dataset loaded from data files.
    _array_order : str
        The order of array dimensions, typically 'CHW' or 'HWC'.

    Methods
    -------
    scale_based_on_spectral_pattern(channels, scale_type='standardization', update_data=False)
        Scales the data of specified channels based on their spectral pattern.
    calculate_color_space(rgbchannels=['red', 'green', 'blue'], color_space='cielab', update_data=True)
        Converts the dataset to the specified color space.
    calculate_vegetation_indices(vi_list=None, vi_equations=None, verbose=False, update_data=True, **kwargs)
        Calculates specified vegetation indices for the dataset.
    """
    @property
    def _available_vi(self):
        from .general import MSVEGETATION_INDEX
        return list(MSVEGETATION_INDEX.keys())
    @property
    def _available_color_spaces(self):

        return {'cielab': ['l','a','b'],'hsv': ['h','s','v']}
    
    @property    
    def _list_color_features(self):
        featcolor = []
        for i in self._available_color_spaces.keys():
            featcolor += self._available_color_spaces.get(i)
        
        return featcolor
    
    def __init__(self, xrdata: xarray.Dataset, array_order: str = 'CHW') -> None:
        """
        Initializes the DataCubeMetrics instance with an xarray dataset and an array order.

        Parameters
        ----------
        xrdata : xr.Dataset
            The xarray dataset that contains the data.
        array_order : str, optional
            Specifies the order of array dimensions, by default 'CHW'.
        """
        self.xrdata = xrdata
        self._array_order = array_order
        
    @abstractmethod
    def _clip_cubedata_image(self, **kwargs):
        """
        Abstract method to be implemented for clipping cube data images.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments for clipping parameters.
        """
        
        pass
    
    @abstractmethod
    def _update_params(self):
        """
        Abstract method to update internal parameters based on the current dataset.
        """
        pass 
    
    @abstractmethod
    def _scale_xrdata(self, **kwargs ):
        """
        Abstract method to be implemented for scaling cube data images.

        Parameters
        ----------

        """
        
        pass
       
    def scale_based_on_spectral_pattern(self, channels, scale_type = 'standardization', update_data = False):
        """
        Scales the data of specified channels based on their spectral pattern, either performing standardization or normalization.

        Parameters
        ----------
        channels : List[str]
            A list of channel names to be scaled.
        scale_type : str, optional
            The type of scaling to apply. Currently, 'standardization', 'normalization', and 'scale' are implemented. Defaults to 'standardization'.
        update_data : bool, optional
            If True, the `xrdata` attribute of the class is updated with the scaled data. Defaults to False.

        Returns
        -------
        xr.Dataset
            The xarray dataset containing the scaled data for the specified channels.

        Notes
        -----
        - This method assumes that `self.custom_dict` contains the dataset in a dictionary format and that
        `self.to_array` and `from_dict_toxarray` methods are implemented for data conversion.
        - Only 'standardization' scaling is currently supported.
        """
        
        if scale_type not in ['standardization','normalization', 'scale']:
            raise NotImplementedError(f"{scale_type} scaling is not supported.")

        datacubedict = from_xarray_to_dict(self.xrdata)
        
        mltdata = get_data_from_dict(datacubedict, onlythesechannels = channels)
        # Determine data layout
        channelfirst = self._array_order == 'CHW'
        datareshape = mltdata.reshape(mltdata.shape[0],mltdata.shape[2]*mltdata.shape[1]).swapaxes(0,1) if channelfirst else mltdata.reshape(mltdata.shape[2]*mltdata.shape[1], mltdata.shape[0])

        if scale_type == 'standardization':
            meanval = np.nanmean(datareshape, axis = 1)
            stdval = np.nanstd(datareshape, axis = 1)
            imgstandardized = np.array([(datareshape[:,i] - meanval)/stdval for i in range(datareshape.shape[1])])
        if scale_type == 'scale':
            stdval = np.nanstd(datareshape, axis = 1)
            imgstandardized = np.array([(datareshape[:,i])/stdval for i in range(datareshape.shape[1])])
            
        if scale_type == 'normalization':
            maxval = np.nanmax(datareshape, axis = 1)
            minval = np.nanmin(datareshape, axis = 1)
            imgstandardized = np.array([(datareshape[:,i] - minval)/(maxval-minval) for i in range(datareshape.shape[1])])

        
        if channelfirst:
            imgstandardized = imgstandardized.reshape(mltdata.shape[0],*mltdata.shape[1:])
        else:
            imgstandardized = imgstandardized.reshape(mltdata.shape[0],*mltdata.shape[1:]).swapaxes(0,1).swapaxes(1,2)
        
        # Update each channel in the data cube dictionary
        for i, chan in enumerate(channels):
            if channelfirst:
                datacubedict['variables'][chan] = imgstandardized[i]
            else:
                datacubedict['variables'][chan] = imgstandardized[:, :, i]

        updated_xrdata = from_dict_toxarray(datacubedict, dimsformat = self._array_order)
                
        if update_data:
            self.xrdata = updated_xrdata
        
        return updated_xrdata
    
    def calculate_color_space(self,
                              rgbchannels: List[str] = ['red','green','blue'],
                              color_space: str = 'cielab',
                              update_data = True
                              ):
        """
        Converts the RGB data within the dataset to the specified color space (CIE LAB or HSV).

        Parameters
        ----------
        rgb_channels : List[str], optional
            List of channel names representing RGB. Defaults to ['red', 'green', 'blue'].
        color_space : str, optional
            The target color space for conversion ('cielab' or 'hsv'). Defaults to 'cielab'.
        update_data : bool, optional
            If True, updates the `xrdata` attribute with the converted dataset. Defaults to True.

        Returns
        -------
        xr.Dataset
            The xarray dataset containing the data converted to the specified color space.
        """
        
        funcolorspace = XRColorSpace(color_space=color_space)
        
        datacubecolor = funcolorspace.transform(self.xrdata, 
                                                rgb_channels=rgbchannels,
                                                array_order=self._array_order)
        if update_data:
            self.xrdata = datacubecolor
            
        return datacubecolor
        
    
    def calculate_vegetation_indices(self, 
                                     vi_list: List[str] = None, vi_equations:dict = None, 
                                     verbose:bool = False, 
                                     update_data = True,
                                     **kwargs):
        """
        Calculate vegetation indices from the clipped data.

        Parameters:
        -----------
        vi_list : List[str], optional
            List of vegetation indices to calculate. Defaults to ['ndvi'].
        vi_equations : Dict[str, str], optional
            Dictionary containing equations for vegetation indices. Defaults to None.
        verbose : bool, optional
            If True, print progress messages. Defaults to False.
        update_data : bool, optional
            If True, update the data with the calculated vegetation indices. Defaults to True.

        Returns:
        --------
        xarray.Dataset
            Dataset containing the calculated vegetation indices.
            
        Notes
        -----
        - The function updates the instance's xrdata attribute if `update_data` is True.
        """
        if vi_list is None:
            vi_list = ['ndvi']
        if vi_equations is None:
            from drone_data.utils.general import MSVEGETATION_INDEX
            vi_equations = MSVEGETATION_INDEX
    
        xrdatac = self.xrdata.copy()
        
        for vi in vi_list:
            if verbose:
                print('Computing {}'.format(vi))
            xrdatac = calculate_vi_fromxarray(xrdatac,vi = vi,
                                              expression = vi_equations[vi], 
                                              **kwargs)
            
        if update_data:
            self.xrdata  = xrdatac
            self._update_params()
        
        return xrdatac
    


class DataCubeReader(CustomXarray):
    
    """
    An abstract base class for reading data cubes from a specific path

    Attributes
    ----------
    path : Optional[str]
        The path to the directory containing data cube files.
    
    Methods
    -------

    read_individual_data(file, path, dataformat)
        Reads individual data from a file and converts it to an xarray dataset.
    listcxfiles
        Retrieves a list of filenames ending with 'pickle' in the specified path.
    """
             
    
    @property
    def listcxfiles(self) -> Optional[List[str]]:
        """
        Retrieves a list of filenames ending with 'pickle' in the specified directory path.

        Returns
        -------
        Optional[List[str]]
            List of filenames ending with 'pickle', or None if path is not set.
        """
        
        if self.path is not None:
            assert os.path.exists(self.path) ## directory soes nmot exist
            files = [i for i in os.listdir(self.path) if i.endswith('pickle')]
        else:
            files = None
        return files
    
    def read_individual_data(self, file: str = None, path: str = None, 
                             dataformat: str = 'CHW') -> dict:
        """
        Read individual data from a file.

        Parameters:
        -----------
        file : str, optional
            Name of the file to read. Defaults to None.
        path : str, optional
            Path to the file directory. Defaults to None.
        dataformat : str, optional
            Data oder format. Defaults to 'CHW'.

        Returns:
        --------
        dict
        """
        if path is not None:
            file = os.path.basename(file)
        else:
            path = self.path
            file = [i for i in self.listcxfiles if i == file][0]
        
        self._array_order = dataformat
        customdict = self._read_data(path=path, 
                                   fn = file,
                                   suffix='pickle')
        
        self.xrdata  = from_dict_toxarray(customdict, dimsformat = dataformat)
        
        return customdict
            
    
    def __init__(self, path: Optional[str] = None, **kwargs) -> None:
        
        """
        Initializes the `DataCubeReader` instance with the specified path and additional arguments.

        Parameters
        ----------
        path : Optional[str], optional
            The path to the directory containing data files. Defaults to None.
        **kwargs : dict
            Additional keyword arguments passed to the `CustomXarray` parent class.
        """
        
        self.xrdata = None
        self.path = path
        
        super().__init__(**kwargs)


          
class DataCubeProcessing(DataCubeMetrics):
    """
    Extends `DataCubeMetrics` to provide specific functionalities like clipping based on an area or buffer
    and scaling data based on spectral patterns for data cube processing.

    Methods
    -------
    _update_params():
        Updates the channel names and number of dimensions from the current dataset.
    _clip_cubedata_image(min_area, buffer, update_data):
        Clips the dataset based on a minimum area or buffer and optionally updates the dataset.
    _scale_xrdata(scaler_values, scale_type, update_data):
        Applies scaling to the dataset based on specified scalar values and type.
    """
    def __init__(self, xrdata: xarray.Dataset, array_order: str = 'CHW') -> None:
        """
        Initializes the DataCubeProcessing instance with an xarray dataset and array order.

        Parameters
        ----------
        xrdata : xr.Dataset
            The xarray dataset loaded from data files.
        array_order : str, optional
            The order of the array dimensions, typically 'CHW', 'HWC', 'DCHW' . Defaults to 'CHW'.
        """
        super().__init__(xrdata=xrdata, array_order=array_order)
        
    
    def _update_params(self):
        """Updates the channel names and number of dimensions based on the current xarray dataset."""
        
        self._channel_names = None if not self.xrdata else list(self.xrdata.keys())
        self._ndims = None if not self.xrdata else len(list(self.xrdata.sizes.keys()))
        
    def _clip_cubedata_image(self, min_area:float =None, 
                             buffer: float = None, 
                             update_data: bool = True):
        """
        Clips the dataset based on a minimum area or buffer distance. If `update_data` is True,
        the internal dataset is updated with the clipped version.

        Parameters
        ----------
        min_area : Optional[float], optional
            The minimum area required for the data cube polygon. If specified, the dataset
            is clipped to ensure the minimum area. Defaults to None.
        buffer : Optional[float], optional
            The buffer distance to apply when clipping the dataset. Defaults to None.
        update_data : bool, optional
            Whether to update the internal dataset with the clipped version. Defaults to True.

        Returns
        -------
        xr.Dataset
            The clipped xarray dataset.

        Raises
        ------
        ValueError
            If neither `min_area` nor `buffer` is provided.
        """
        
        # using area or buffer
        if min_area is not None:
            
            datacubepolygon = get_xarray_polygon(self.xrdata)
            if datacubepolygon.area < min_area:
                return self.xrdata

            [xmin,xmax],[ymin,ymax] = get_minmax_pol_coords(datacubepolygon)
            
            buffergeom = estimate_pol_buffer([xmin,xmax],[ymin,ymax],min_area)
            buffer = buffergeom/2
            
            
        elif buffer is None:
            raise ValueError("Please provide either min_area or buffer.")
        
        datageom = gpd.GeoDataFrame(geometry=[datacubepolygon])
        
        clippedxrdata  =clip_xarraydata(self.xrdata.copy(), 
                                    datageom.loc[:,'geometry'], 
                                    buffer = buffer)
        if update_data:
            self.xrdata = clippedxrdata
            
        return clippedxrdata

    def _scale_xrdata(self, scaler_values: Dict[int, List[float]], 
                      scale_type: str = 'standardization',
                      update_data: bool = True):
        """
        Applies scaling to a list of numpy arrays based on the specified scaling type.
        
        Parameters
        ----------
            
        scaler : dict
            Dictionary that contains the scalar values per channel. 
            e.g. for example to normalize the red channel you will provide min and max values {'red': [1,255]}  
        scale_sype:  str, optional
            String to mention if 'standarization' or 'normalization' is gonna be applied. Defaults to 'standarization'.

        """
        if self.xrdata:
            scaledxrdata = xr_data_transformation(self.xrdata,scaler= scaler_values, scalertype=scale_type)
        elif self._customdict:
            scaledxrdata = customdict_transformation(self.xrdata,scaler= scaler_values, scalertype=scale_type)
        
        if update_data:
            self.xrdata = scaledxrdata
        
        return scaledxrdata


def getting_only_rgb_channels(mltarray, features, rgb_channels = ['red','green','blue']):

    rgbarray = []
    for i, band in enumerate(rgb_channels):
        #
        for bpos in range(len(features)):
            if band == features[bpos]:
                rgbarray.append(mltarray[bpos])

    rgbarray = np.stack(rgbarray, axis=0)
    
    return rgbarray
    
def insert_rgb_to_matrix(origarray, rgbarray, features, rgb_channels = ['red','green','blue']):
    
    newarray = origarray.copy()
    for i, band in enumerate(rgb_channels):
        for bpos in range(len(features)):
            if band == features[bpos]:
                newarray[bpos] = rgbarray[i]
    
    return newarray

 

def clip_datacube(datacube: xarray.Dataset, min_area: Optional[float] = None, 
                  image_reduction: Optional[float] = None) -> xarray.Dataset:
    """
    Clips the data cube based on a specified minimum area or a percentage reduction of the current area.

    Parameters
    ----------
    datacube : xarray.Dataset
        The data cube to clip, assumed to have an 'xrdata' attribute.
    min_area : float, optional
        The minimum area to retain after clipping.
    image_reduction : float, optional
        The percentage by which to reduce the current area.

    Returns
    -------
    Dataset
        The clipped data cube.
    """
    # clip either by a given minum area threshold or using a image reduciton factor
    if min_area is not None:
        current_area = get_xarray_polygon(datacube.xrdata).area
        if current_area> min_area:
            datacube._clip_cubedata_image(min_area = min_area,update_data=True)
    if image_reduction is not None and image_reduction>0:
        current_area = get_xarray_polygon(datacube.xrdata).area
        reduced_area = current_area * (1-image_reduction)
        datacube._clip_cubedata_image(min_area = reduced_area,update_data=True)
    
    return datacube
            
def create_new_features(datacube: xarray.Dataset, features: List[str], 
                        rgb_for_color_space: List[str] = ['red', 'green', 'ms']):
    """
    Creates new features in the data cube, such as vegetation indices and color spaces.

    Parameters
    ----------
    datacube : xarray.Dataset
        The data cube for feature creation.
    features : List[str]
        List of features to consider for creation.
    rgb_for_color_space : List[str], optional
        RGB channel names to use for color space calculations.
    """
    # calculate vegetation indices 
    vilist = [i for i in datacube._available_vi if i in features]
    if len(vilist)>0:
        datacube.calculate_vegetation_indices(vilist, overwrite = True)

    # calculate colors
    colorlist = [color for color in datacube._list_color_features if color in features]
    if colorlist:
        colospaces = np.unique([i for j in colorlist for i in datacube._available_color_spaces.keys() if j in datacube._available_color_spaces[i]])
        for i in colospaces:
            datacube.calculate_color_space(color_space = i, 
                                                rgbchannels =rgb_for_color_space, 
                                                update_data=True)
            
def resize_mlt_data(img_data: np.ndarray, newsize, 
                    interpolation: str = 'bicubic') -> np.ndarray:
    
    """
    Resizes multi-layer data to a new size using specified interpolation.

    Parameters
    ----------
    img_data : np.ndarray
        The image data to resize, expected to have shape (channels, depth, height, width).
    newsize : List[int, int]
        The new size as (height, width).
    interpolation : str, optional
        Interpolation method to use (e.g., 'bicubic', 'bilinear').

    Returns
    -------
    np.ndarray
        The resized multi-dimensional data.
    """
    ## resize data
    mltdataarray = np.zeros((img_data.shape[:2]+tuple(newsize)))
    for c in range(img_data.shape[0]):
        for d in range(img_data.shape[1]):
            mltdataarray[c,d] = resize_2dimg(img_data[c,d], newsize[0], newsize[1], flip=False, 
                                                interpolation = interpolation, blur= False)

    return mltdataarray

def apply_image_augmentation(image: np.ndarray, transformation: str, channel_names: List[str], 
                             rgb_channels: List[str], tr_configuration: Optional[Dict] = None, 
                             trverbose: bool = False, transformoptions: List[str] = ['rotation', 'flip']) -> np.ndarray:
    """
    Applies specified image augmentation to the provided multi-dimensional image data.

    Parameters
    ----------
    image : np.ndarray
        The image data to augment.
    transformation : str
        The type of transformation to apply.
    channel_names : List[str]
        Names of the channels in the image data.
    rgb_channels : List[str]
        Specific channels to apply RGB-based transformations.
    tr_configuration : dict, optional
        Configuration for transformations.
    trverbose : bool, optional
        If true, enables verbose output during transformation.
    transformoptions : List[str], optional
        Available transformation options.

    Returns
    -------
    np.ndarray
        The augmented image data.
    """
    try:
        from Crop_CV.dataset.imagery_transformation import MultiTimeTransform
    except:
        print("There is not a module called Crop_CV, please donwloaded first to apply image augmentation")
        transformation = 'raw'
        
    data = image
    if transformation == 'illumination':
    ## chcking if rgb is in the array
        imgrgb = getting_only_rgb_channels(image,channel_names, rgb_channels)
        if imgrgb.shape[0] == 3:
            mltdataaug = MultiTimeTransform(data=imgrgb , formatorder="CDHW", transform_options = tr_configuration)          
            data = mltdataaug.random_multime_transform(verbose = trverbose, augfun='illumination')
            data = insert_rgb_to_matrix(image, data, channel_names,rgb_channels)
        
    else:
        mltdataaug = MultiTimeTransform(data=image , formatorder="CDHW")
        data = mltdataaug.random_multime_transform(verbose = trverbose, augfun = transformation)
    
    return data



class MultiDDataTransformer(DataCubeProcessing):
    """
    Processes depth imagery data with multiple channels by applying transformations,
    scaling, and other data manipulations.

    Attributes
    ----------
    channels : List[str]
        List of channel names from the data cube.
    transformations : Optional[Any]
        Transformation options for processing the image data.
    scaler : Dict
        Scaling parameters for image data normalization or standardization.
    time_points : Optional[List[int]]
        Specific time points to slice from the data cube.
    
    Methods
    -------
    get_transformed_image
        Applies specified transformations and returns the processed image data.
    """
    def __init__(self, xrdata: Optional[xarray.Dataset] = None, array_order: str = 'CHW', 
                 transformation_options: Optional[str] = None, 
                 channels: Optional[List[str]] = None,
                 time_points: Optional[List[int]] = None, 
                 scaler: Optional[Dict[str, float]] = None) -> None:
        
        """
        Initializes the DepthImageryTranformation class with data cube and transformation details.

        Parameters
        ----------
        xrdata : Dataset, optional
            The xarray Dataset containing the data cube.
        array_order : str, optional
            The order of array dimensions.
        transformation_options : Any, optional
            Options for transforming the image data.
        channels : List[str], optional
            Specific channels to use from the data cube.
        time_points : List[int], optional
            Time points to extract from the data cube.
        scaler : Dict[str, Any], optional
            Parameters for data scaling.
        """
        
        self.channels = list(xrdata.keys()) if channels is None else channels
        self.transformations = transformation_options
        self.scaler = scaler
        self.time_points = time_points
        super().__init__(xrdata, array_order)
    
    def get_transformed_image(self, 
                              min_area: float = None, 
                              image_reduction: float = None, 
                              augmentation: str = None,
                              rgb_for_color_space: List[str] = ['red', 'green', 'ms'],
                              rgb_for_illumination: List[str] = ['red', 'green', 'ms'],
                              new_size: Optional[int] = None, 
                              scale_rgb: bool = True) -> np.ndarray:
        """
        Processes the imagery data by applying transformations and returning the modified image data.

        Parameters
        ----------
        min_area : float, optional
            Minimum area for data clipping.
        image_reduction : float, optional
            Reduction factor for clipping based on image area.
        augmentation : str, optional
            Name of the augmentation to apply.
        rgb_for_color_space : List[str], optional
            Channels to use for color space calculations.
        rgb_for_illumination : List[str], optional
            Channels to use for illumination adjustments.
        new_size : int, optional
            New size for resizing the image data.
        scale_rgb : bool, optional
            Flag to scale RGB channels.

        Returns
        -------
        np.ndarray
            The processed multi-channel image data.
        """
        
        # clip datacube
        self.clip_datacube(min_area=min_area, image_reduction = image_reduction)
        # create new features in t given case that there are channels that are not in the datacube
        self.create_new_features(rgb_for_color_space = rgb_for_color_space)
        # get the array data
        mltdata = self.to_4darray(new_size)
        # data augmentation
        if augmentation is None or augmentation not in self.transformations:
            trfunction = np.random.choice(self.transformations) if self.transformations is not None else 'raw'
        else:
            trfunction = augmentation
        #print(mltdata.shape, trfunction)
        mltdata = self.tranform_mlt_data(mltdata, transformation=trfunction, rgb_channels=rgb_for_illumination)
        # data standarization
        rgbchannelsonimage = [i for i in ['red','green','blue'] if i in self.channels]
        if scale_rgb and len(rgbchannelsonimage)>0:
            imgrgb = getting_only_rgb_channels(mltdata,self.channels,rgbchannelsonimage)
            imgrgb = imgrgb / 255. if np.max(imgrgb)>1 else imgrgb
            mltdata = insert_rgb_to_matrix(mltdata, imgrgb, self.channels, rgb_channels=rgbchannelsonimage)
            
        if self.scaler is not None:
            mltdata = self.scale_mlt_data(mltdata)
            
        return mltdata
    
    def clip_datacube(self, min_area: Optional[float] = None, image_reduction: Optional[float] = None) -> None:
        """
        Clips the data cube based on a minimum area threshold or an image reduction factor.

        Parameters
        ----------
        min_area : float, optional
            Minimum area required for the data cube to retain after clipping.
        image_reduction : float, optional
            Factor by which to reduce the data cube area.

        Notes
        -----
        Either `min_area` or `image_reduction` should be specified to perform clipping.
        """
        # clip either by a given minum area threshold or using a image reduciton factor
        if min_area is not None:
            current_area = get_xarray_polygon(self.xrdata).area
            if current_area> min_area:
                self._clip_cubedata_image(min_area = min_area,update_data=True)
        if image_reduction is not None and image_reduction>0:
            current_area = get_xarray_polygon(self.xrdata).area
            reduced_area = current_area * (1-image_reduction)
            self._clip_cubedata_image(min_area = reduced_area,update_data=True)
        
    
    def create_new_features(self, rgb_for_color_space: List[str] = ['red', 'green', 'ms']) -> None:
        """
        Creates new features such as vegetation indices and color spaces based on available data channels.

        Parameters
        ----------
        rgb_for_color_space : List[str], optional
            RGB channel names used for color space calculations.

        Notes
        -----
        Adds calculated features directly to the dataset, updating it in-place.
        """
        
        # calculate vegetation indices 
        vilist = [i for i in self._available_vi if i in self.channels]
        if len(vilist)>0:
            self.calculate_vegetation_indices(vilist, overwrite = True)

        # calculate colors
        featcolor = self._list_color_features
        colorlist = [i for i in featcolor if i in self.channels]
        if len(colorlist)>0:
            colospaces = np.unique([i for j in colorlist for i in self._available_color_spaces.keys() if j in self._available_color_spaces[i]])
            for i in colospaces:
                self.calculate_color_space(color_space = i, 
                                                    rgbchannels =rgb_for_color_space, 
                                                    update_data=True)
    
    def tranform_mlt_data(self, imagedatacube, transformation: Optional[str] = None, 
                          rgb_channels: List[str] = ['red', 'green', 'blue'], 
                          tr_configuration = None):
        """
        Transforms the multi-dimensional data cube by applying specified image augmentation.

        Parameters
        ----------
        imagedatacube : np.ndarray
            The image data cube to transform.
        transformation : str, optional
            The type of transformation to apply (defaults to 'raw' if None).
        rgb_channels : List[str], optional
            List of RGB channel names involved in the transformation.

        Returns
        -------
        np.ndarray
            The transformed image data cube, with CDHW dimension order.
        """
        transformation = 'raw' if transformation is None else transformation
        data = apply_image_augmentation(imagedatacube, transformation, self.channels, rgb_channels, 
                                 tr_configuration = tr_configuration)
        
        return data
    
    def scale_mlt_data(self, data: np.ndarray) -> np.ndarray:
        """
        Scales the multi-dimensional data cube using the provided scaler configuration.

        Parameters
        ----------
        data : np.ndarray
            Data to be scaled.

        Returns
        -------
        np.ndarray
            The scaled data cube.
        """
        data = transform_listarrays(data, var_channels = self.channels, scaler = self.scaler, 
                                    scale_type ='standardization')
        
        data = np.array([data[chan] for chan in list(data.keys())])
        if True in np.isnan(data):
            data = fill_na_values(data, n_neighbors = 7)
        return data    
    
    def to_4darray(self, new_size: Optional[int] = None) -> np.ndarray:
        """
        Converts the xarray data set (datacube) into a 4-dimensional array.

        Parameters
        ----------
        new_size : int, optional
            Desired size for each image dimension if resizing is needed.

        Returns
        -------
        np.ndarray
            The converted or resized 4-dimensional array.
        """
        lendims = len(list(self.xrdata.sizes.keys()))
        
        if self.time_points is not None and lendims >2:
            self.xrdata = self.xrdata.isel(date = self.time_points)
            
        data = get_data_from_dict(from_xarray_to_dict(self.xrdata), onlythesechannels = self.channels)
        if len(data.shape) == 3:
            data = np.expand_dims(data, axis = 1)

        if new_size is not None:
            data = resize_mlt_data(data, [new_size,new_size], interpolation = 'bicubic')
        
        return data

        

    