import os
import pickle
import numpy as np
import random
from image_utils.transform import Augmentation_Xrdata


def minmax_scale(data, minval = None, maxval = None):
    
    if minval is None:
        minval = np.nanmin(data)
    if maxval is None:
        maxval = np.nanmax(data)
    
    return (data - minval) / ((maxval - minval))

def standard_scale(data, meanval = None, stdval = None):
    if meanval is None:
        meanval = np.nanmean(data)
    if stdval is None:
        stdval = np.nanstd(data)
    
    return (data-meanval)/stdval 

def read_xrdata_fromfn(fn, scaler=None, times=None, maskvalues = None):
    
    if os.path.exists(fn):
        with open(fn,"rb") as f:
            xrdata= pickle.load(f)

    else:
        raise ValueError(f'there is no a file with this name {fn}')
    xrdatamod = xrdata.copy()

    if scaler is not None:
        scale1, scale2,method = scaler
        
        for varname in list(xrdatamod.keys()):
            if method == 'minmax':
                xrdatamod[varname] = minmax_scale(
                    xrdatamod[varname],
                    minval = scale1[varname], 
                    maxval = scale2[varname])
            else:
                xrdatamod[varname] = standard_scale(
                    xrdatamod[varname],
                    meanval = scale1[varname], 
                    stdval = scale2[varname])
    
    if maskvalues is not None:
        xrdatamod = xrdatamod.where(np.logical_not(np.isnan(xrdatamod)), 0)
    
    if times is not None:
        xrdatamod = xrdatamod.isel(date = times)

    return xrdatamod

class GettingDLDatafromXRdata(object):
    

    def __get_randomtransform(self, idx):
        #imgpath = imgpath[:imgpath.index('_time')]
        fn = os.path.join(self.path,self.fn_list[idx])
        xrdata = read_xrdata_fromfn(fn, self._scaler_image_values, self._dates, maskvalues = 0)
        
        augxrdata = Augmentation_Xrdata(xrdata, variables=self._features)

        #if self._img_transform:
        res = augxrdata.random_multime_transform(augfun=self._aug_transforms)

        return res
        
    def __init__(self, 
                 path,
                 fn_list,
                 target_Values,
                 scaler_target_values = None,
                 scaler_image_values = None,
                 features = None,
                 times = None,
                 augmentation_options = None):

        #super().__init__()
        self.path = path
        self.fn_list = fn_list
        self._target = target_Values
        self._channelslast = True
        self._features = features
        self._dates = times
        self._scaler_image_values = scaler_image_values
        self._scaler_target = scaler_target_values
        
        self._aug_transforms = augmentation_options

    def __len__(self):
        return len(self.fn_list)

    def __getitem__(self,idx):
        x = self.__get_randomtransform(idx)
        if self._channelslast:
            x = x.swapaxes(1,2).swapaxes(2,3)

        y = self._target[idx]

        if self._scaler_target is not None:
            y  = self.__normalize(y)

        return x, [y]
    
    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            
            if i == self.__len__()-1:
                self.on_epoch_end()
    
    def __normalize(self, x):
        return (x-self._scaler_target[0])/self._scaler_target[1]        

    #shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        reidx = random.sample(population = list(range(self.__len__())),k = self.__len__())
        self.fn_list = self.fn_list[reidx]

