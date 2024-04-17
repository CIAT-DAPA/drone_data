


import json
import os
import inspect
import pandas as pd
import numpy as np


from typing import Tuple, Optional, List, Any

import torch
from torch.utils.data import Dataset


from ..ml_utils.general_functions import SplitIdsClassification
from ..utils.datacube_transforms import MultiDDataTransformer, DataCubeReader
from ..utils.general import split_filename

class TargetDataset():
    """
    A class to manage datasets containing target variables, with support for loading data from a CSV file and retrieving non-NaN target values.

    Parameters
    ----------
    phase : str
        Specifies the dataset phase, 'train' or 'validation'.
    path : str, optional
        The file path to the dataset CSV file. If not specified, no file will be loaded.
    target_key : str, optional
        The column name in the dataset that contains the target values.
    id_key : str, optional
        The column name that contains identifiers for each data point.
    tabletype : str, optional
        The type of data structure to load the data into, default is 'dataframe'.

    Attributes
    ----------
    train : bool
        Whether this dataset is for training.
    validation : bool
        Whether this dataset is for validation.
    file_path : str
        The path to the dataset file.
    target_label : str
        The column name for target values.
    ids_label : str
        The column name for data identifiers.
    _df : pd.DataFrame
        The loaded dataset as a DataFrame.
    """
    
    def __init__(self, phase, path:str=None, target_key:str = None, id_key:str = None, table_type = 'dataframe') -> None:

        
        self.train = phase == 'train' 
        self.validation = phase == 'validation'
                
        self.file_path = path
        if path is not None:
            assert os.path.exists(self.file_path), f"The path {path} does not exist"

        self.target_label = target_key
        self.ids_label = id_key
        
        #self.target_transformation = parser.scaler_transformation
        if table_type == "dataframe":
            self._df = pd.read_csv(self.file_path) 
       #if tabletype == "dict":

       
    def _non_nan_positions(self):
        """
        Finds positions of non-NaN entries in the target column.

        Returns
        -------
        List[int]
            Indices of non-NaN entries in the target column.
        """
        target = self._df[self.target_label].values
        return [i for i,value in enumerate(target) if not np.isnan(value)]
        
    
    def get_ids(self):
        """
        Retrieves the identifiers for data points with non-NaN target values.

        Returns
        -------
        List[int]
            A list of identifiers corresponding to non-NaN target values.
        """
        nnpos = self._non_nan_positions()
        if self.ids_label in self._df.columns:
            ids = list(self._df[self.ids_label].values[nnpos])
        else:
            ids = list(range(nnpos))
        return ids
    
    
    def get_target(self):
        """
        Retrieves the target values that are non-NaN.

        Returns
        -------
        np.ndarray
            Array of non-NaN target values.
        """
        nnpos = self._non_nan_positions()
        
        target = self._df[self.target_label].values[nnpos]    
        self._orig_target = target
        
        return target
    
class ClassificationTarget(TargetDataset, SplitIdsClassification):
    
    """
    Handles dataset operations for classification tasks including data retrieval and stratified data splitting.

    Inherits from:
    - TargetDataset for basic data handling.
    - SplitIdsClassification for stratified splitting of IDs based on target values.

    Methods
    -------
    split_data(cv=None, nkfolds=None)
        Splits the data into training and validation sets based on the provided cross-validation setup or number of folds.
    get_target_value(index)
        Retrieves the target value and corresponding ID for the specified index.
    """
    
    def __init__(self, **kwargs) -> None:
        """
        Initializes the ClassificationTarget with data handling and splitting capabilities.

        Parameters are inherited from TargetDataset and SplitIdsClassification, including:
        - phase (str): Specifies if the dataset is used for training or validation.
        - path (str, optional): Path to the dataset CSV file.
        - target_name, id_colname, tabletype: Dataset specific configurations.
        - targetvalues (np.ndarray): Array of target classification values.
        - ids, val_perc, test_perc, seed, shuffle, testids_fixed, stratified: Data splitting parameters.
        """
        parameters = inspect.signature(TargetDataset.__init__).parameters
        print(self.get_params_fromwargs(parameters,kwargs))
        
        TargetDataset.__init__(self, **self.get_params_fromwargs(parameters,kwargs))
        
        
        parameters = inspect.signature(SplitIdsClassification.__init__).parameters
        SplitIdsClassification.__init__(self, targetvalues=self.get_target(), **self.get_params_fromwargs(parameters,kwargs))

    @property
    def ids_data(self):
        return self.get_ids()
    
    @property
    def target_data(self):
        return self.get_target()
    
    #def get_params_fromwargs(class)
    
    def split_data(self,  cv: Optional[int] = None, nkfolds: Optional[int] = None) -> Tuple[List[Any], List[Any]]:
        """
        Splits the data for training or validation using either cross-validation indices or a number of folds.

        Parameters
        ----------
        cv : Optional[int]
            The specific cross-validation split index to use.
        nkfolds : Optional[int]
            Number of folds if using k-fold splitting.

        Returns
        -------
        Tuple[List[Any], List[Any]]
            A tuple containing lists of IDs and corresponding target data for the split.
        """
        # getting training or validation ids
        if self.train and not self.validation:
            if cv is None:

                idstosplit = self._get_new_stratified_ids(self._initial_tr_ids)
            else:
                idstosplit = self.stratified_kfolds(nkfolds)[cv][0]
        elif self.validation:
            if cv is None:
                idstosplit = self._get_new_stratified_ids(self._initial_test_ids)
            else:
                idstosplit = self.stratified_kfolds(nkfolds)[cv][1]
        else:
            idstosplit = self._get_new_stratified_ids(self._initial_test_ids)
            
        idsdata = [self.ids_data[i] for i in idstosplit]
        
        trdata = [self.target_data[i] for i in idstosplit]
        
        self._idssubset = idsdata
        self._valuesdata = trdata
        return [idsdata, trdata]
    
    
    def get_target_value(self, index: int) -> Tuple[str, float]:
        """
        Retrieves the target value and its corresponding ID based on the provided index.

        Parameters
        ----------
        index : int
            Index for which to retrieve the target and ID.

        Returns
        -------
        Tuple[Any, Any]
            The ID and target value at the specified index.
        """
        return [self._idssubset[index], self._valuesdata[index]]
        
    @staticmethod
    def get_params_fromwargs(classparams,kwargs):
        """
        Extracts parameters for class initialization from arguments passed to the constructor.

        Parameters
        ----------
        classparams : Dict[str, Any]
            Parameters expected by the class constructor.
        kwargs : Dict[str, Any]
            Arguments provided to the constructor.

        Returns
        -------
        Dict[str, Any]
            Filtered dictionary of parameters applicable to the class constructor.
        """
        
        parameters = {key: value for key, value in classparams.items() if key not in ['self', 'args', 'kwargs']}
        parameterstarget = {key: kwargs.get(key) for key in kwargs.keys() if key in list(parameters.keys())}
        
        return parameterstarget 


class ClassificationData(Dataset,DataCubeReader,ClassificationTarget):
    def _check_filename(self, filename):
        """
        Check if the provided filename includes a directory path and update class variables accordingly.

        Parameters
        ----------
        filename : str
            The file name or path to split.

        Returns
        -------
        None
        """
        path, fn = split_filename(filename)
        self._tmppath = path
        self._file = fn

    
    def __len__(self):
        """
        Return the total number of items in the dataset.

        Returns
        -------
        int
            Total number of items.
        """
        
        return len(self._idssubset)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves an item by its index for model training/testing.

        Parameters
        ----------
        index : int
            Index of the item to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the image tensor and its corresponding target tensor.
        """
        
        idimg, targetval = self.get_target_value(index)
        self._check_filename(idimg)

        self.read_individual_data(
                    path = self._tmppath, file=self._file,  dataformat = self.confi['input_array_order'])
        
        ## activate tranform module
        datacube_metrics = MultiDDataTransformer(self.xrdata, 
                                            transformation_options =self.confi['transform_options'],
                                            channels=self.confi['feature_names'].split('-'),
                                            scaler= self._scalar_values)
        
        datatransformed = datacube_metrics.get_transformed_image(min_area = self.confi['minimun_area'], 
                                    image_reduction=self.confi['image_reduction'],
                                    #augmentation='raw',
                                    new_size= self.confi['input_image_size'],
                                    rgb_for_color_space = self.confi['rgb_channels_for_color_space'],
                                    rgb_for_illumination = self.confi['rgb_channels_for_illumination'])
        
        imgtensor = torch.from_numpy(datatransformed.swapaxes(0,1)).float()
        
        targetval = np.array(targetval)
        targetten = torch.from_numpy(np.expand_dims(targetval, axis=0)).float()
        
        if imgtensor.shape[1] == 1:
            imgtensor = torch.squeeze(imgtensor)
        
        return imgtensor, targetten
        
    def __init__(self, configuration_dict: dict)-> None : 
        """
        Initializes the classification data handler with configurations.

        Parameters
        ----------
        configuration_dict : dict
            Configuration dictionary specifying operational parameters.
        """
        self.confi = configuration_dict
        
        ClassificationTarget.__init__(self,**self.confi)

        cv = None if self.confi["cv"] == -1 else self.confi["cv"]
        nkfolds = None if self.confi['kfolds'] == 0 else self.confi['kfolds']
        
        self.split_data(cv = cv, nkfolds = nkfolds)
        
        DataCubeReader.__init__(self)
        
        if os.path.exists(self.confi['scaler_path']):
            with open(self.confi['scaler_path'], 'rb') as fn:
                self._scalar_values = json.load(fn)


