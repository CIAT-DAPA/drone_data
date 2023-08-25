

import os
from sklearn.metrics import f1_score
import json
import numpy as np
import pandas as pd

class EvaluateSuffix(object):
    @staticmethod
    def _check_json_suffix(fn):
        #assert fn[0](fn)
        return fn.endswith('.json')
            
    def __init__(self, arg) -> None:
        self._arg = arg
        
    def __call__(self, *args):
        
        if len(args) == 1:
            fn = args[0]
        else:
            fn = args[1]

        out =  self._arg(fn) if self._check_json_suffix(fn) else None
        return out
    
@EvaluateSuffix
def loadjson(fn):
    """
    Load JSON data from a file.

    Parameters
    ----------
    fn : str
        Filename of the JSON file to load.

    Returns
    -------
    dict or None
        Dictionary containing the loaded JSON data.
        Returns None if the file does not exist.
    """
    
    if os.path.exists(fn):
        with open(fn, "rb") as fn:
            reporter = json.load(fn)
    else:
        reporter = None
    return reporter

class ClassificationReporter(object):
    """
    A class for managing and analyzing classification report data.

    Methods
    -------
    update_reporter(new_entry)
        Update the reporter with a new entry.
    load_reporter(fn)
        Load the reporter data from a JSON file.
    scores_summary(scorenames='cvscores')
        Calculate the summary of a score metric.
    best_score(scorenames='cvscores')
        Retrieve the best score from the reporter data.
    save_reporter(fn)
        Save the reporter data to a JSON file.

    Attributes
    ----------
    reporter : dict
        Dictionary containing the classification report data.
    _reporter_keys : list of str
        List of reporter keys.
    """
    
    def reset_reporter(self):
        reporter = {}
        for keyname in self._reporter_keys:
            reporter.update({keyname: []})
        
        self.reporter = reporter
        
    def update_reporter(self, new_entry):    
        """
        Update the reporter with a new entry.

        Parameters
        ----------
        new_entry : dict
            Dictionary containing the new entry to add.

        Returns
        -------
        None
        """
        
        for k in list(self._reporter_keys):
            self.reporter[k].append(new_entry[k])        
    
    def load_reporter(self, fn):    
        reporter = loadjson(fn)
        if reporter is None:
            print('s')
            reporter = {}
            for keyname in self._reporter_keys:
                reporter.update({keyname: []})
        else:
            print('load')
        
        self.reporter = reporter
    
    def scores_summary(self, scorenames = 'cvscores'):
        return [np.mean(score) for score in self.reporter[scorenames]]
    
    def best_score(self, scorenames = 'cvscores'):
        
        orderedpos = np.argsort(self.scores_summary(scorenames))
        
        rout = {}
        for keyname in self._reporter_keys:
            rout[keyname] = self.reporter[keyname][orderedpos[-1]]
        
        return rout
    
    def save_reporter(self, fn):
        json_object = json.dumps(self.reporter, indent=4)
        with open(fn, "w") as outfile:
            outfile.write(json_object)
    
    def __init__(self, _reporter_keys = None) -> None:
        
        if _reporter_keys is None:
            self._reporter_keys = ['features','cvscores']
        else:
            self._reporter_keys = _reporter_keys
            

class DL_ClassReporter(ClassificationReporter):
    def n_models(self):
        breaks = self._get_breaks()
        return len(breaks)-1
    
    def _get_breaks(self):
        splitpos = []
        for j,i in enumerate(self.reporter[self.iterationcolumn]):
            if i == 0:
                splitpos.append(j)
        if len(splitpos) == 1:
            splitpos.append(len(self.reporter[self.iterationcolumn]))
        return splitpos
    
    def get_data(self,index):
        breaks = self._get_breaks()
        dicttmp = {}
        for i in self.reporter.keys():        
            dicttmp[i] = self.reporter[i][breaks[index]:breaks[index+1]]
            
        return dicttmp
    
    def pickupmax_performance(self, nmodelid, evalmetric, maxvalue = True):
        data = self.get_data(nmodelid)
        if maxvalue:
            pos = np.argmax(data[evalmetric])
        else:
            pos = np.argmin(data[evalmetric])
        # extracting data for that position
        dicttmp = {k: data[k][pos] for k in list(self.reporter.keys())}

        return dicttmp
    
    def summarize_cv_restults(self, evalmetric, cvcolumn = 'cv', featuresnames = "features", groupbycolumns = None):
        pdsumm = pd.DataFrame(self.summarize_all_models_bymax(evalmetric=evalmetric))
        
        if type(pdsumm[featuresnames][0]) == list:
            pdsumm[featuresnames]= pdsumm[featuresnames].apply(lambda x: '-'.join(x))
        
        if groupbycolumns is not None:
            grouped = pdsumm.groupby(groupbycolumns)
        else:
            grouped = pdsumm.groupby(featuresnames)
        
        #set minimun cv number
        cvn = np.max(np.unique(self.reporter[cvcolumn]))
        pdsummf = grouped.filter(lambda x: x.shape[0] > cvn)
        if groupbycolumns is not None:
            grouped = pdsummf.groupby(groupbycolumns)
        else:
            grouped = pdsummf.groupby(featuresnames)
            
        return grouped[evalmetric].mean().reset_index().sort_values(evalmetric,ascending=False)
    

    def summarize_all_models_bymax(self, evalmetric, **kwargs):
        assert evalmetric in list(self.reporter.keys())
        valueslist = []
        
        for i in range(self.n_models()):    
            dicttmp = self.pickupmax_performance(i,evalmetric, **kwargs)
            valueslist.append(dicttmp)
        return valueslist
    
    def unique_attribute(self,attributes, evalmetric):
        
        summaryres =self.summarize_all_models_bymax(evalmetric).copy()
        datafeatunique =[]
        for iterresult in summaryres:
            attr = [iterresult[attr] for attr in attributes]
            if attr not in datafeatunique:
                datafeatunique.append(attr) 

        return datafeatunique
    
    def __init__(self, _reporter_keys=None, iterationcolumn = 'iteration') -> None:
        super().__init__(_reporter_keys)
        self.iterationcolumn = iterationcolumn



def concatenate_lists(listoflist):
    
    idsfromgroup = [] 
    for j in range(len(listoflist)):
        listunique = []
        for i in listoflist[j]:
            listunique.append('-'.join(i) if type(i) is list else i)
        idsfromgroup.append('-'.join(listunique))
    return(idsfromgroup)

class CVReporter(object):
    """
    A class to summarize cross-validation classification or regression results .

    Methods
    -------
    unique_attributes(new_entry)
        select shunks of .
    load_reporter(fn)
        Load the reporter data from a JSON file.
    scores_summary(scorenames='cvscores')
        Calculate the summary of a score metric.
    best_score(scorenames='cvscores')
        Retrieve the best score from the reporter data.
    save_reporter(fn)
        Save the reporter data to a JSON file.

    Attributes
    ----------
    reporter : dict
        Dictionary containing the classification report data.
    _reporter_keys : list of str
        List of reporter keys.
    """
    def unique_attributes(self):
        
        assert len(self.groupby) == sum([i in self.reporter._reporter_keys for i in self.groupby])
        
        datafeatunique =[]
        for i in range(len(self.reporter.reporter[self.groupby[0]])):
            attr = [self.reporter.reporter[attr][i] for attr in self.groupby]
            if attr not in datafeatunique:
                datafeatunique.append(attr)
    
        return datafeatunique
        

    def cv_groups(self, maxfeatures = None):
        
        uniqueresults = self.unique_attributes()
        uniqueresults = concatenate_lists(uniqueresults)
        groupsbycv = []
        for i in uniqueresults:
            # get attributes
            indvals = []
            for j in range(len(self.reporter.reporter[self.groupby[0]])):
                
                compared = concatenate_lists(
                    [self.reporter.reporter[featname][j] if type(self.reporter.reporter[featname][j]) is list 
                     else [self.reporter.reporter[featname][j]] for featname in self.groupby ])
                if '-'.join(compared) == i:
        
                    indvals.append({featname:self.reporter.reporter[featname][j] 
                                    for featname in list(self.reporter.reporter.keys())})
            
            groupsbycv.append(indvals)
        
        if maxfeatures is not None:
            groupsbycv = [i for i in groupsbycv if len(i[0]['features']) == maxfeatures]

        return groupsbycv

    def cv_summary(self, eval_metric = 'valaccuracy', maxfeatures = None):
        
        cvgroups = self.cv_groups(maxfeatures = maxfeatures)
        cvvals = []
        for cvgroup in cvgroups:
            cvval = np.mean([cvval[eval_metric] for cvval in cvgroup])
            dictvalues = {i:cvgroup[0][i] for i in self.groupby}
            dictvalues.update({eval_metric:cvval})
            cvvals.append(dictvalues)
        
        return cvvals
    
    def best_result(self, eval_metric = 'valaccuracy', maxfeatures = None):
        cvgroups = self.cv_summary(eval_metric = eval_metric, maxfeatures = maxfeatures)
        
        return cvgroups[np.argmax([i[eval_metric] for i in cvgroups])]
    
    def __init__(self, reporter, groupby = None) -> None:
        
        self.reporter = reporter
        
        groupby = groupby if type(groupby) is list else [groupby]
        self.groupby = groupby