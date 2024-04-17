import numpy as np

import random
from sklearn.model_selection import KFold,StratifiedKFold
import os
import pandas as pd
import numpy as np
from itertools import compress
import itertools

import copy

from typing import List, Optional, Union

def select_columns(df, colstoselect, additionalfilt = 'gr_'):
    colsbol = [i.startswith(colstoselect[0]
    ) and additionalfilt not in i for i in df.columns]

    for cols in range(1, len(colstoselect)):
        colsbol = np.array(colsbol) | np.array([i.startswith(
            colstoselect[cols]) and additionalfilt not in i for i in df.columns])

    return df[list(compress(df.columns,  colsbol))]

def split_idsintwo(ndata, ids = None, percentage = None, fixedids = None, seed = 123):
    """
    Split the IDs into two sets.

    Args:
        ids_length (int): Length of the IDs.
        ids (list): List of IDs.
        percentage (float): Percentage of data to allocate into one group.
        fixedids (list): List of IDs that can be used to split the data.
        seed (int): Random seed.

    Returns:
        tuple: A tuple containing both groups of ids.
    """
    if ids is None:
        ids = list(range(len(ndata)))

    if percentage is not None:
        if fixedids is None:
            idsremaining = pd.Series(ids).sample(int(ndata*percentage), random_state= seed).tolist()
        else:
            idsremaining = fixedids
        
        main_ids = [i for i in ids if i not in idsremaining]
    elif fixedids is not None:
        idsremaining = fixedids
        main_ids = [i for i in ids if i not in idsremaining]
    
    else:
        idsremaining = None
        main_ids = ids

    return main_ids, idsremaining


def retrieve_datawithids(data, ids):
    if len(ids) > 0:
        subset  = data.iloc[ids]
    else:
        subset = None

    return subset

def split_dataintotwo(data, idsfirst, idssecond):

    subset1 = data.iloc[idsfirst]
    subset2 = data.iloc[idssecond]

    return subset1, subset2


class SplitIds(object):

    
    def _ids(self):
        """
        Generate a list of IDs ranging from 0 to (ids_length - 1).

        Returns:
            list: A list of IDs.
        """
        
        ids = list(range(self.ids_length))
        if self.shuffle:
            ids = pd.Series(ids).sample(n = self.ids_length, random_state= self.seed).tolist()

        return ids


    def _split_test_ids(self, test_perc):
        self.training_ids, self.test_ids = split_idsintwo(self.ids_length, self.ids, test_perc,self.test_ids, self.seed)


    def kfolds(self, kfolds, shuffle = True):
        kf = KFold(n_splits=kfolds, shuffle = shuffle, random_state = self.seed)

        idsperfold = []
        for train, test in kf.split(self.training_ids):
            idsperfold.append([list(np.array(self.training_ids)[train]),
                               list(np.array(self.training_ids)[test])])

        return idsperfold
    
    def __init__(self, ids_length: Optional[int] = None, ids = None,val_perc =None, test_perc = None,seed = 123, shuffle = True, testids_fixed = None) -> None:
        """
        Initializes the SplitIds class for splitting data into different sets.

        Parameters:
        ----------
        ids_length : int, optional
            The total number of observations. Required if 'ids' is not provided.
        ids : list, optional
            A list of unique identifiers. Required if 'ids_length' is not provided.
        val_perc : float, optional
            The proportion of data to be used for the validation set (0.0 to 1.0).
        test_perc : float, optional
            The proportion of data to be used for the test set (0.0 to 1.0).
        seed : int, optional
            Seed for random number generation to ensure reproducibility.
        shuffle : bool, optional
            Whether to shuffle the IDs before splitting.
        testids_fixed : list, optional
            A predefined list of IDs to be used as the test set.

        Raises:
        ------
        ValueError
            If neither 'ids_length' nor 'ids' are provided or if the percentages are out of range.
        """
        
        
        self.shuffle = shuffle
        self.seed = seed
        
        if ids is None and ids_length is not None:
            self.ids_length = ids_length
            self.ids = self._ids()
        elif ids_length is None and ids is not None:
            self.ids_length = len(ids)
            self.ids = ids
        else:
            raise ValueError ("provide an index list or a data length value")
        
        self.val_perc = val_perc

        if testids_fixed is not None:
            self.test_ids = [i for i in testids_fixed if i in self.ids]
        else:
            self.test_ids = None

        self.training_ids, self.test_ids = split_idsintwo(self.ids_length, self.ids, test_perc,self.test_ids, self.seed)
        if val_perc is not None:
            self.training_ids, self.val_ids = split_idsintwo(len(self.training_ids), self.training_ids, val_perc, seed = self.seed)
        else:
            self.val_ids = None
        

class SplitIdsClassification(SplitIds):
    """
    Extends SplitIds to support stratified splitting based on classification targets, including stratified K-fold splitting.

    Attributes
    ----------
    targetvalues : np.ndarray
        Array of target classification values corresponding to each data point.
    categories : np.ndarray
        Unique categories present in the target values.
    mindataper_category : Optional[int]
        The minimum number of data points present in any category.
    _datapercategory : Dict[str, int]
        A dictionary mapping category to the number of data points in that category.
    
    Methods
    -------
    countdata_percategory(targetvalues)
        Counts the number of data points per category.
    stratified_kfolds(kfolds, shuffle)
        Generates stratified K-folds for the provided target values.
    """
    
    def countdata_percategory(self, targetvalues):
        """
        Counts the number of data points per category in the target values.

        Parameters
        ----------
        targetvalues : np.ndarray
            Target classification values for each data point.
        """
        
        listperc = {}
        for i in range(len(self.categories)):
            datapercat = np.sum(targetvalues == self.categories[i])
            #print(datapercat)
            listperc[str(int(self.categories[i]))] =  datapercat#(datapercat/ len(targetvalues))
            #listperc[str(int(self.categories[i]))] =datapercat
            
        self._datapercategory = listperc   
    
    def _get_mindata(self):
        """
        Finds the category with the minimum number of data points.

        Returns
        -------
        Tuple[int, str]
            The minimum number of data points in any category and the corresponding category.
        """
        mindata = len(self.targetvalues)
        mincat = -1
        for i in list(self._datapercategory.keys()):
            if self._datapercategory[i]<mindata:
                mindata = self._datapercategory[i]
                mincat = i
                
        return mindata,mincat 
    
    def _get_new_stratified_ids(self,listids: List[int], 
                                seed: int = 123) -> List[int]:
        """
        Generates new stratified ids based on the minimum data per category.

        Parameters
        ----------
        listids : List[int]
            List of ids to stratify.
        seed : int, optional
            Random seed for reproducibility, by default 123.

        Returns
        -------
        List[int]
            A list of new stratified ids.
        """
        stratids = []
        #print(self.mindataper_category)
        tmpcat = self.targetvalues[listids]
        self.countdata_percategory(tmpcat)
        self.mindataper_category,mincat = self._get_mindata()
        
        for i in range(len(self.categories)):
            catvalues = np.array(listids)[tmpcat == self.categories[i]]
            df = pd.DataFrame({'ids':catvalues})
            if str(int(self.categories[i])) != mincat:
                nsample = int(self.mindataper_category) #* df.shape[0])
            else:
                nsample = df.shape[0]

            stratids.append(df.sample(n=nsample, random_state=seed)['ids'].values)
        
        return list(itertools.chain.from_iterable(stratids))
    
    
    def stratified_kfolds(self, kfolds: int, shuffle: bool = True) -> List[List[List[int]]]:
        """
        Generates stratified K-folds for cross-validation.

        Parameters
        ----------
        kfolds : int
            Number of folds for the cross-validation.
        shuffle : bool, optional
            Whether to shuffle the data before splitting into folds, by default True.

        Returns
        -------
        List[List[List[int]]]
            A list of folds, each containing training and validation ids.
        """
        
        #if self.mindataper_category:        
        kf = StratifiedKFold(n_splits=kfolds, shuffle = shuffle, random_state = self.seed)
        
        stratifiedids = [self._get_new_stratified_ids(
            self.training_ids.copy(), seed=self.seed+(i*10)) for i in range(kfolds)]
        
        i = 0
        idsperfold = []
        for i,(train, test) in enumerate(kf.split(np.array(stratifiedids[i]),
                                                            np.array(self.targetvalues)[stratifiedids[i]])):
            
            idsperfold.append([list(np.array(stratifiedids[i])[train]),
                                        list(np.array(stratifiedids[i])[test])])
            
        #else:
        #    idsperfold = self.kfolds(kfolds)
        
        return idsperfold
            
    
    def __init__(self, 
                 targetvalues: np.ndarray, 
                 ids: Optional[List[int]] = None, 
                 val_perc: Optional[float] = None, 
                 test_perc: Optional[float] = None, 
                 seed: int = 123, 
                 shuffle: bool = True, 
                 testids_fixed: Optional[List[int]] = None,
                 stratified: bool = True) -> None:
        
        """
        Initializes the SplitIdsClassification instance.

        Parameters
        ----------
        targetvalues : np.ndarray
            Target classification values for each data point.
        ids : Optional[List[int]], optional
            List of ids representing data points, by default None.
        val_perc : Optional[float], optional
            Percentage of data to allocate to validation set, by default None.
        test_perc : Optional[float], optional
            Percentage of data to allocate to test set, by default None.
        seed : int, optional
            Random seed for reproducibility, by default 123.
        shuffle : bool, optional
            Whether to shuffle data before splitting, by default True.
        testids_fixed : Optional[List[int]], optional
            List of ids to always include in the test set, by default None.
        stratified : bool, optional
            Whether the split should be stratified, by default True.
        """
        
        self.targetvalues = targetvalues
        self.categories = np.unique(targetvalues)
        super().__init__(len(targetvalues), ids, val_perc, test_perc, seed, shuffle, testids_fixed)
        
        self._initial_tr_ids = copy.deepcopy(self.training_ids)
        self._initial_val_ids = copy.deepcopy(self.val_ids)
        self._initial_test_ids = copy.deepcopy(self.test_ids)
        self.mindataper_category = None
        self._datapercategory = {}
            

class SplitData(object):
    """
    A class for managing data splits into test, training, and validation sets, as well as supporting K-fold splitting.

    Attributes
    ----------
    data : Any
        The complete dataset from which subsets will be extracted.
    ids_partition : Any
        An object containing attributes `test_ids`, `training_ids`, `val_ids`, and a method `kfolds()` for K-fold splitting.
    kfolds : Optional[int]
        The number of K-folds for cross-validation. If None, K-fold splitting is not used.

    Methods
    -------
    kfold_data(kifold)
        Returns training and validation data for the specified K-fold.
    """
    
    @property
    def test_data(self):
        """Retrieves the test subset of the data."""
        return retrieve_datawithids(self.data, self.ids_partition.test_ids) 
    
    @property
    def training_data(self):
        """Retrieves the training subset of the data."""
        return retrieve_datawithids(self.data, self.ids_partition.training_ids) 
    
    @property
    def validation_data(self):
        """Retrieves the validation subset of the data."""
        return retrieve_datawithids(self.data, self.ids_partition.val_ids) 

    def kfold_data(self, kifold):
        """
        Returns training and validation data subsets for the specified K-fold index.

        Parameters
        ----------
        kifold : int
            The index of the fold for which to retrieve the data.

        Returns
        -------
        Tuple[Optional[Any], Optional[Any]]
            A tuple containing the training and validation datasets for the specified fold.
            Returns (None, None) if K-folds are not defined or if the fold index is out of range.
        """
        tr, val = None, None
        if self.kfolds is not None:
            if kifold <= self.kfolds:
                tr, val = split_dataintotwo(self.data, 
                                            idsfirst = self.ids_partition.kfolds(self.kfolds)[kifold][0], 
                                            idssecond = self.ids_partition.kfolds(self.kfolds)[kifold][1])

        return tr, val
        
    def __init__(self, df, splitids, kfolds = None) -> None:
        """
        Initializes the SplitData instance with the dataset, an object for ID partitions, and an optional K-fold count.

        Parameters
        ----------
        df : Any
            The complete dataset.
        split_ids : Any
            An object containing ID partitions for splitting the data.
        kfolds : Optional[int], optional
            The number of folds for K-fold cross-validation, by default None.
        """
        self.data = df
        self.ids_partition = splitids
        self.kfolds = kfolds




def get_cv_folds_index(data_length, nfolds=None, seed=1990, 
                      shuffle=True, test_prc = None, 
                      testids = None):
    test_data= None
    ### obtain the cv indexes for a group of data
    totallbels = np.arange(0, data_length, dtype=int)
    
    
    if shuffle:
        random.seed(seed)
        random.shuffle(totallbels)
    if testids is not None:
        test_data = np.array(testids)
        totallbels = np.array([i for i in totallbels if i not in testids])    

    elif test_prc is not None:
        testlength = int(data_length*(test_prc/100))
        test_data = totallbels[(len(totallbels) - testlength):]
        totallbels = totallbels[:(len(totallbels) - testlength)]  
    else:
        test_data = []
    
    if nfolds is not None:
        random.seed(seed)    
        kf = KFold(n_splits=nfolds)
        id_perfold = []
        for train, test in kf.split(totallbels):
            id_perfold.append([totallbels[train],
                            totallbels[test]])
        
        id_perfold= [id_perfold,test_data]

    else:
        id_perfold = [[[totallbels, test_data]],[]]    

    return id_perfold


def get_id_test_train(data_length, seed=1990, shuffle=True, 
                      test_prc = None, testids = None,
                      validation = True, val_perc = 10):
    test_data= None
    ### obtain the cv indexes for a group of data
    totallbels = np.arange(0, data_length, dtype=int)
    
    
    if shuffle:
        random.seed(seed)
        random.shuffle(totallbels)

    if testids is not None:
        test_data = np.array(testids)
        totallbels = np.array([i for i in totallbels if i not in testids])  
    
    elif test_prc is not None:
        testlength = int(data_length*(test_prc/100))
        test_data = totallbels[(len(totallbels) - testlength):]
        totallbels = totallbels[:(len(totallbels) - testlength)]

    else:
        test_prc = 20
        testlength = int(data_length*(test_prc/100))
        test_data = totallbels[(len(totallbels) - testlength):]
        totallbels = totallbels[:(len(totallbels) - testlength)]

    id_perfold= [[totallbels,test_data]]
    ## if there is also validation
    if validation:
        validationlength = int(len(totallbels)*(val_perc/100))
        validationids = totallbels[(len(totallbels) - validationlength):]
        totallbels = totallbels[:(len(totallbels) - validationlength)]
        id_perfold= [[[totallbels,validationids]], test_data]
    

    return id_perfold


class SplitData(object):

    def __init__(self,
        lendata,
        kfolds = None, 
        val_prct = 10, 
        seed = 123,
        test_indexes = None,
        validation = True
        ):

        
        self.val_data = None
        self.cv_indexes = None
        self.kfolds = kfolds
        self.cv_results = None
        

        if self.kfolds is not None:
            self.cv_indexes, self.test_indexes = get_cv_folds_index(
                                lendata,
                                nfolds = self.kfolds,
                                seed = seed,
                                test_prc=val_prct, testids=test_indexes)

        elif validation:
            
            trindex = get_id_test_train(lendata,
                                seed = seed,
                                test_prc=val_prct,
                                testids=test_indexes,
                                validation=validation,
                                val_perc=val_prct)

            self.test_indexes = trindex[1]
            self.cv_indexes = trindex[0]

        else:
            
            trindex = get_id_test_train(lendata,
                                seed = seed,
                                test_prc=val_prct,
                                testids=test_indexes,
                                validation=validation,
                                val_perc=val_prct)

            self.test_indexes = trindex[0][1]
            self.cv_indexes = trindex


def split_dataset(ids,cv_indexes):
    ### get cross validation training and validation indexes
    
    return [[np.array(ids)[
        cv_indexes[i][0]],np.array(ids)[cv_indexes[i][1]]] for i in range(len(cv_indexes))]
   


class SplitDatabyids(SplitData):

    @property
    def test_ids(self):
        return np.array(self.fn)[list(self.test_indexes)]

    @property
    def training_ids(self):
        return split_dataset(self.fn,self.cv_indexes)
    
    def __init__(self, 
        ids,
        kfolds = None, 
        val_prct = 10, 
        seed = 123,
        test_ids = None,
        validation = True) -> None:

        self.fn = ids
        
        super().__init__(len(ids),
        kfolds = kfolds,         
        val_prct = val_prct, 
        seed = seed,
        test_indexes=test_ids,
        validation = validation)
        