import copy
import geopandas as gpd
import pandas as pd
import math
import xarray
import torch
import torch.optim as optim
import cv2
import numpy as np
import random
import os
import tqdm

from typing import List, Tuple, Optional

from ..utils.general import euclidean_distance
from ..utils.xr_functions import crop_xarray_using_mask, CustomXarray, from_dict_toxarray


def _apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image



def getmidleheightcoordinates(pinit,pfinal,alpha):

  xhalf=math.sin(alpha) * euclidean_distance(pinit,pfinal)/2 + pinit[0]
  yhalf=math.cos(alpha) * euclidean_distance(pinit,pfinal)/2 + pinit[1]
  return int(xhalf),int(yhalf)

def getmidlewidthcoordinates(pinit,pfinal,alpha):

  xhalf=pfinal[0] - math.cos(alpha) * euclidean_distance(pinit,pfinal)/2
  yhalf=pinit[1] - math.sin(alpha) * euclidean_distance(pinit,pfinal)/2
  return int(xhalf),int(yhalf)


#
def get_heights_and_widths(maskcontours):

    p1,p2,p3,p4=maskcontours
    alpharad=math.acos((p2[0] - p1[0])/euclidean_distance(p1,p2))

    pheightu=getmidleheightcoordinates(p2,p3,alpharad)
    pheigthb=getmidleheightcoordinates(p1,p4,alpharad)
    pwidthu=getmidlewidthcoordinates(p4,p3,alpharad)
    pwidthb=getmidlewidthcoordinates(p1,p2,alpharad)

    return pheightu, pheigthb, pwidthu, pwidthb


def get_boundingboxfromseg(mask):
    
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    
    return([xmin, ymin, xmax, ymax])

def get_height_width(grayimg, pixelsize: float = 1) -> Tuple[float, float]:
    """
    Calculate the height and width of an object in an image.

    Parameters:
    -----------
    grayimg : np.ndarray
        Grayscale input image.
    pixelsize : float, optional
        Size of a pixel. Defaults to 1.

    Returns:
    --------
    Tuple[float, float]
        Tuple containing the height and width of the object.
    """
    
    from Crop_CV.dataset.image_functions import find_contours
    
    ## find contours
    wrapped_box = find_contours(grayimg)

    ## get distances
    pheightu, pheigthb, pwidthu, pwidthb = get_heights_and_widths(wrapped_box)
    d1 = euclidean_distance(pheightu, pheigthb)
    d2 = euclidean_distance(pwidthu, pwidthb)
    
    pheightu, pheigthb, pwidthu, pwidthb = get_heights_and_widths(wrapped_box)
    d1 = euclidean_distance(pheightu, pheigthb)
    d2 = euclidean_distance(pwidthu, pwidthb)

    ## with this statement there is an assumption that the rice width is always lower than height
    larger = d1 if d1>d2 else d2
    shorter = d1 if d1<d2 else d2
    
    return larger* pixelsize, shorter*pixelsize


          
class SegmentationDataCube(CustomXarray):
    """
    A class for handling data cube extending CustomXarray functionality,
    allowing operations for segmentation such as clipping based on mask layers and listing specific files.
    """
    
    @property
    def listcxfiles(self):
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
    
    
    def _clip_cubedata_image(self, min_threshold_mask:float = 0, padding: int = None):
        """
        Clips the cube data image based on the bounding box of the masking layer.

        Parameters
        ----------
        min_threshold_mask : float, optional
            Minimum threshold value for the mask. Defaults to 0.
        padding : Optional[int], optional
            Padding size. Defaults to None.

        Returns
        -------
        np.ndarray
            The clipped image as a numpy array.
        """
        ## clip the maksing layer 
        #maskbb = get_boundingboxfromseg(self._maskimg*255.)
        clipped_image = crop_xarray_using_mask(self._maskimg*255., self.xrdata, 
                                               buffer = padding,min_threshold_mask = min_threshold_mask)

        return clipped_image
        
    def _check_mask_values(self, mask_name, change_mask = False):
        """
        Checks and adjusts the mask image values based on predefined criteria.
        
        Adjusts the mask image by selecting an alternative mask if the sum is 0,
        squeezing the array if it has an unnecessary third dimension, and scaling
        the values if the maximum is not above a threshold.
        """
        if np.nansum(self._maskimg) != 0:
            if change_mask and len(self._msk_layers)>1:
                mask_name= [i for i in self._msk_layers if i != mask_name][0]
                self._maskimg = self.to_array(self._customdict,onlythesechannels = [mask_name])
            
        self._maskimg = self._maskimg.squeeze() if len(self._maskimg.shape) == 3 else self._maskimg
        self._maskimg = self._maskimg if np.max(self._maskimg) > 10 else self._maskimg*255.
        
    
    def mask_layer_names(self,
                         mask_suffix: str =  'mask',
                         ): 
        """
        Selects mask layer names from the cube data based on a suffix.

        Parameters
        ----------
        mask_suffix : str, optional
            Suffix to filter mask layers by. Defaults to 'mask'.

        Returns
        -------
        List[str]
            List of mask layer names.

        Raises
        ------
        ValueError
            If no mask layers are found.
        """
        # selcet mask name
        varnames = list(self.xrdata.keys())
        mask_layer_names = [i for i in varnames if i.startswith(mask_suffix)] if mask_suffix else None
        if not mask_layer_names:
            raise ValueError("There is no mask")
        
        self._msk_layers = mask_layer_names
        return mask_layer_names
    
    def clip_using_mask(self, 
                        mask_name: str = None, 
                        channels: List[str] = None,
                        padding: int = 0,
                        paddingincm: bool = False,
                        mask_data: bool = False,
                        mask_value: float = 0,
                        min_threshold_mask : float = 0):
        
        """
        Clip data using a specified mask.

        Parameters:
        -----------
        mask_name : str, optional
            Name of the mask. Defaults to None.
        channels : List[str], optional
            List of channels to clip. Defaults to None.
        padding : int, optional
            Padding size. Defaults to 0.
        paddingincm : bool, optional
            The padding size is in centimeters. Defaults to False
        mask_data : bool, optional
            Use the mask layer to mask the final datacube 
        min_threshold_mask : float, optional
            Minimum threshold value for the mask. Defaults to 0.
        Returns:
        --------
        np.ndarray
            Clipped image array.
        """
    
        channels = list(self.xrdata.keys()) if channels is None else channels
        # select mask name
        if self._msk_layers is not None:
            mask_name = random.choice(self._msk_layers) if mask_name is None else mask_name
        
        assert mask_name is not None
        
        # get data mask as array       
        self._maskimg = self.xrdata[mask_name].values# to_array(self._customdict, onlythesechannels = [mask_name])
        self._check_mask_values(mask_name)
        # padding in pixels
        if np.nansum(self._maskimg) != 0:
            padding =  int(padding/(self.xrdata.attrs['transform'][0]*100)) if paddingincm else padding
            
            ## clip the xarray            
            clipped_image = self._clip_cubedata_image(min_threshold_mask, padding) if np.nansum(self._maskimg) > 0 else self.xrdata.copy()

            if mask_data:
                clipped_image = clipped_image.where(clipped_image[mask_name]>min_threshold_mask,mask_value)
        else:
            clipped_image = None
            
        self._clippeddata = clipped_image
        
        return clipped_image
    
    
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
        
        self._arrayorder = dataformat
        customdict = self._read_data(path=path, 
                                   fn = file,
                                   suffix='pickle')
        
        self.xrdata  = from_dict_toxarray(customdict, dimsformat = dataformat)
        
        return customdict
            
        
    def __init__(self, path: Optional[str] = None, **kwargs) -> None:
        """
        Initializes the CubeDataMetrics instance with the specified path and additional arguments.

        Parameters
        ----------
        path : Optional[str], optional
            The path to the directory containing data files, by default None.
        **kwargs : dict
            Additional keyword arguments passed to the CustomXarray parent class.
        """
        self.xrdata = None
        self._msk_layers = None
        self.path = path
        super().__init__(**kwargs)
        