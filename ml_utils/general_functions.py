import numpy as np

import random
from sklearn.model_selection import KFold


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
        