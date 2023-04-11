

import os
from sklearn.metrics import f1_score
import json
import numpy as np


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
    if os.path.exists(fn):
        with open(fn, "rb") as fn:
            reporter = json.load(fn)
    else:
        reporter = None
    return reporter

class ClassificationReporter(object):
    
    def update_reporter(self, new_entry):    
        for k in list(self._reporter_keys):
            self.reporter[k].append(new_entry[k])        
    
    def load_reporter(self, fn):    
        reporter = loadjson(fn)
        if reporter is None:
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
        