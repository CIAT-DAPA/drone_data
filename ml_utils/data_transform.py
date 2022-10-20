import os
import pickle
import numpy as np
import random
from image_utils.transform import AugmentMTdata
import copy
from utils.image_functions import transform_to_heightprofile
import keras
import concurrent.futures as cf
import ray
from itertools import product


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

def read_xrdata_fromfn(fn, 
                       scaler=None, 
                       times=None, 
                       maskvalues = None,
                       scale_z = True):
    
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
                if scale_z and varname == 'z':
                        xrdatamod[varname] = minmax_scale(
                            xrdatamod[varname],
                            minval = scale1[varname], 
                            maxval = scale2[varname])
                elif varname != 'z':
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
import multiprocessing



def run_tsheightparallel(args):
    
	#print(inputargs[0][1])
    #return ray.get([operation.remote(fn_list[i],path, 
    #                            scalerimage, 
    #                            features,
    #                            dates,
    #                            heightvarname,
    #                            heightprofile, transforms) for i in listidx])
    j = args[0]
    data = args[1]
    #print(data[j])
    return transform_to_heightprofile(data[j],data[j])

def height_profiles_atime(ts2dheight, zscaler = None,parallel =None,
                          **kwargs):

    if parallel=='startmap':
        print(len(list(product(
                list(range(len(ts2dheight))),
                    [ts2dheight]))))
        pool = multiprocessing.Pool(6)
        zprofiles =pool.map(run_tsheightparallel,product(
                list(range(len(ts2dheight))),
                    [ts2dheight]))
        pool.close()
        pool.join()
        print('parallel_f')
    else:
        zprofiles = np.array([
        transform_to_heightprofile(
            ts2dheight[timepoint],
            ts2dheight[timepoint], **kwargs
            ) for timepoint in range(len(ts2dheight))])


    if zscaler is not None:
        scale1, scale2,method = zscaler
        if method == 'minmax':
            zprofiles = minmax_scale(zprofiles,
                        minval = scale1['z'], 
                        maxval = scale2['z'])
        
        else:
            zprofiles = standard_scale(
                        zprofiles,
                        meanval = scale1['z'], 
                        stdval = scale2['z'])

    return zprofiles.swapaxes(1,2).swapaxes(2,3)

import multiprocessing
import time
  
  
class Process(multiprocessing.Process):
    def __init__(self, id):
        super(Process, self).__init__()
        self.id = id
                 
    def run(self):
        time.sleep(1)
        print("I'm the process with id: {}".format(self.id))


#keras.utils.Sequence




class CustomXarray(object):

    def _export_aspickle(self, path, fn, verbose = False) -> None:

        if not os.path.exists(path):
            os.mkdir(path)
        
        outfn = os.path.join(path,fn+'.pickle')
        with open(outfn, "wb") as f:
            pickle.dump(self._filetoexport, f)

        if verbose:
            print('dat exported to {}'.format(outfn))

        
    def export_as_dict(self, path, fn, **kwargs):

        self._filetoexport = self.to_custom_dict()
        self._export_aspickle(path, fn,**kwargs)

    def export_as_pickle(self, path, fn,**kwargs):

        self._filetoexport = self.xrdata
        self._export_aspickle(path, fn,**kwargs)


    def to_custom_dict(self):

        datadict = {
            'variables':{},
            'dims':{},
            'attributes': {}}

        self.variables = list(self.xrdata.keys())
        
        for feature in self.variables:
            datadict['variables'][feature] = self.xrdata[feature].values

        for dim in self.xrdata.dims.keys():
            datadict['dims'][dim] = np.unique(self.xrdata[dim])
        
        for attr in self.xrdata.attrs.keys():
            datadict['attributes'][attr] = '{}'.format(self.xrdata.attrs[attr])

        return datadict

    def __init__(self, xarraydata) -> None:
        
        self.xrdata = xarraydata


def get_heighttsdata(image, heightvarname, features, scaler_image_values):
        
    if heightvarname in features:
        heightpos = features.index(heightvarname)
        heighdata = image.copy().swapaxes(0,1)[heightpos]

        axis = random.choice([0,1])
        heightdata = height_profiles_atime(
            heighdata, axis = axis,
            zscaler =  scaler_image_values,
            flip = True, slices_step=6)
        trimage = copy.deepcopy(image)

        trimage = np.delete(trimage, heightpos, axis=1)
        #self._features = self._features.copy()
        #self._features.pop(heightpos)

    else:
        heightdata = None

    return trimage,heightdata


@ray.remote
def randomtransformfromaug(fn,path, scalerimage, features,dates,heightvarname,heightprofile, transforms):
        #imgpath = imgpath[:imgpath.index('_time')]
        fn = os.path.join(path,fn)
        mtaugdata = AugmentMTdata(fromfile=fn, 
                                removenan= True, 
                                image_scaler=scalerimage, 
                                onlythesefeatures=features,
                                onlythesedates = dates,
                                name_3dfeature= heightvarname,
                                scale_3dimage=np.logical_not(heightprofile))

        #if self._img_transform:
        img_transformed = mtaugdata.random_multime_transform(
            augfun=transforms)

        if heightprofile:
            img_transformed = get_heighttsdata(img_transformed, heightvarname, features, scalerimage)
        else:
            img_transformed = [img_transformed]

        return img_transformed



def randomtransformfromaugnotray(fn,path, scalerimage, features,dates,heightvarname,heightprofile, transforms):
        #imgpath = imgpath[:imgpath.index('_time')]
        fn = os.path.join(path,fn)
        mtaugdata = AugmentMTdata(fromfile=fn, 
                                removenan= True, 
                                image_scaler=scalerimage, 
                                onlythesefeatures=features,
                                onlythesedates = dates,
                                name_3dfeature= heightvarname,
                                scale_3dimage=np.logical_not(heightprofile))

        #if self._img_transform:
        img_transformed = mtaugdata.random_multime_transform(
            augfun=transforms)

        if heightprofile:
            img_transformed = get_heighttsdata(img_transformed, heightvarname, features, scalerimage)
        else:
            img_transformed = [img_transformed]

        return img_transformed

def run_complex_operations(operation, path, fn_list, listidx,scalerimage, 
                           features,dates,heightvarname,heightprofile, transforms):
    
	#print(inputargs[0][1])
    return ray.get([operation.remote(fn_list[i],path, 
                                scalerimage, 
                                features,
                                dates,
                                heightvarname,
                                heightprofile, transforms) for i in listidx])


def one_run_wrapper(*args):
    
    return randomtransformfromaugnotray(*args)

from itertools import product
from multiprocessing import Pool
from multiprocessing import Process
#keras.utils.Sequence
class GettingDLDatafromMTdata(keras.utils.Sequence):

    #from osgeo import gdal
    
    
    def _getheighttsdata(self):
        if self._heightprofile:
            if self._heightvarname in self._features:
                heightpos = self._features.index(self._heightvarname)
                heighdata = self.trimage.copy().swapaxes(0,1)[heightpos]

                axis = random.choice([0,1])
                self.heightdata = height_profiles_atime(
                    heighdata, axis = axis,
                    zscaler =  self._scaler_image_values,
                    flip = True, slices_step=5)
                trimage = copy.deepcopy(self.trimage)

                self.trimage = np.delete(trimage, heightpos, axis=1)
                #self._features = self._features.copy()
                #self._features.pop(heightpos)

            else:
                self.heightdata = None

        else:
            self.heightdata = None

    def _get_randomtransform(self, idx):
        #imgpath = imgpath[:imgpath.index('_time')]
        fn = os.path.join(self.path,self.fn_list[idx])
        mtaugdata = AugmentMTdata(fromfile=fn, 
                                removenan= True, 
                                image_scaler=self._scaler_image_values, 
                                onlythesefeatures=self._features,
                                onlythesedates = self._dates,
                                name_3dfeature= self._heightvarname,
                                scale_3dimage=np.logical_not(self._heightprofile ))

        #if self._img_transform:
        img_transformed = mtaugdata.random_multime_transform(
            augfun=self._aug_transforms)

        self.trimage =  copy.deepcopy(img_transformed)

    def __init__(self, 
                 path,
                 fn_list,
                 target_Values,
                 shuffle = True,
                 scaler_target_values = None,
                 scaler_image_values = None,
                 features = None,
                 times = None,
                 get_zprofiles = False,
                 batch_size=1,
                 heightvariable = 'z',
                 workers = 6,
                 parrallel = 'pool',
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
        self._heightprofile = get_zprofiles
        self._aug_transforms = augmentation_options
        self._heightvarname = heightvariable
        self.heightdata = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._workers = workers
        self.parallel = parrallel
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.fn_list) / self.batch_size))

    def __initrun(self,idx,image):
        #
        if self._channelslast:
            x = image[0].swapaxes(1,2).swapaxes(2,3)
        else:
            x = image[0]
        
        if len(image)> 1:
            x = [x, image[1]]

        y = self._target[idx]

        if self._scaler_target is not None:
            y  = self.__normalize(y)

        return x, y


    def __data_generation(self, list_ids):
        #import concurrent.futures as cf
        

        if self._workers is not None:

            if self.parallel=='starmap':
                pool = multiprocessing.Pool(self._workers)
                
                listdata =pool.starmap(randomtransformfromaugnotray, 
                                        [(self.fn_list[idx],
                                        self.path, 
                                        self._scaler_image_values, 
                                        self._features,
                                        self._dates,
                                        self._heightvarname,
                                        self._heightprofile, self._aug_transforms) 
                                        for idx in list_ids])

                pool.close()
                pool.join()

            if self.parallel=='ray':
                ray.init()
                listdata =run_complex_operations(randomtransformfromaug,
                                                self.path, 
                                                self.fn_list, 
                                                list_ids,
                                                self._scaler_image_values, 
                                                self._features,
                                                self._dates,
                                                self._heightvarname,
                                                self._heightprofile, self._aug_transforms)
                ray.shutdown()

            if self.parallel=='pool':

                with Pool(processes=self._workers) as pool:
                    pool.map(one_run_wrapper, [(self.fn_list[i],
                                self.path, 
                                self._scaler_image_values, 
                                self._features,
                                self._dates,
                                self._heightvarname,
                                self._heightprofile, self._aug_transforms) for i in list_ids])

            #product(list(range(allpolygons.shape[0])),
            #        [[kmeanmodel_soil,5]])

            #with multiprocessing.Pool(processes=self._workers) as pool:
            #    listdata = pool.map(randomtransformfromaug(self.path, 
            #                        self.fn_list, 
            #                        idx,
            #                        self._scaler_image_values, 
            #                        self._features,
            #                        self._dates,
            #                        self._heightvarname,
            #                        self._heightprofile, self._aug_transforms), list_ids)
            # p = Process(target=self.initrun, args=(list_ids))
            #p.start()
            #p.join()    
            #pool = multiprocessing.Pool()
            #pool = multiprocessing.Pool(processes=self._workers)
            #inputs = [0,1,2,3,4]
            #listdata = pool.map(self.initrun, list_ids)
            #pool.close()
            #pool.join()
            #listdata= []
            #cloud_thread= []
            #with cf.ProcessPoolExecutor(max_workers=self._workers) as executor:
            #    for idx in list_ids:
            #        cloud_thread.append({executor.submit(randomtransformfromaug,
            #                                             self.path, 
            #                                                self.fn_list, 
            #                                                idx,
            #                                                self._scaler_image_values, 
            #                                                self._features,
            #                                                self._dates,
            #                                                self._heightvarname,
            #                                                self._heightprofile, self._aug_transforms): idx})

            #    for future in cloud_thread:
            #        for i in cf.as_completed(future):
            #            listdata.append(i.result())
        else:
            listdata = [randomtransformfromaugnotray(self.fn_list[idx],
                                                            self.path, 
                                                            self._scaler_image_values, 
                                                            self._features,
                                                            self._dates,
                                                            self._heightvarname,
                                                            self._heightprofile, self._aug_transforms) for idx in list_ids]
        
        reslist = []
        for idx,img in zip(list_ids,listdata):
            reslist.append(self.__initrun(idx,img))
        ylist = []
        xlist = []
        for xx, yy in reslist:
            xlist.append(xx)
            ylist.append(yy)

        
        if self._heightprofile:
            ximages = np.array([values[0] for values in xlist])
            xprofiles = np.array([values[1] for values in xlist])
            xlist = [ximages,xprofiles]
        else:
            xlist = np.array(xlist)
        
        return xlist, np.array(ylist)


    def __getitem__(self,idx):
        
        indexes = self._list_ids[idx*self.batch_size:(idx+1)*self.batch_size]

        # Find list of IDs
        x, y = self.__data_generation(indexes)

        return x, y
    
    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            
            if i == self.__len__()-1:
                self.on_epoch_end()
    
    def __normalize(self, x):
        return (x-self._scaler_target[0])/self._scaler_target[1]        

    #shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        self._list_ids = list(range(len(self.fn_list)))
        
        if self.shuffle == True:
            np.random.shuffle(self._list_ids)


"""

class GettingDLDatafromXRdata(object):

    #from osgeo import gdal
    
    
    def _getheighttsdata(self):
        if self._heightprofile:
            if self._heightvarname in self._features:
                heightpos = self._features.index(self._heightvarname)
                heighdata = self.trimage.copy().swapaxes(0,1)[heightpos]

                axis = random.choice([0,1])
                self.heightdata = height_profiles_atime(
                    heighdata, axis = axis,
                    zscaler =  self._scaler_image_values,
                    flip = True, slices_step=5)
                trimage = copy.deepcopy(self.trimage)

                self.trimage = np.delete(trimage, heightpos, axis=1)
                #self._features = self._features.copy()
                #self._features.pop(heightpos)

            else:
                self.heightdata = None

        else:
            self.heightdata = None

    def _get_randomtransform(self, idx):
        #imgpath = imgpath[:imgpath.index('_time')]
        fn = os.path.join(self.path,self.fn_list[idx])
        xrdata = read_xrdata_fromfn(fn, self._scaler_image_values, self._dates, maskvalues = 0,
                                    scale_z = np.logical_not(self._heightprofile ))
        
        augxrdata = Augmentation_Xrdata(xrdata, 
                                        variables=self._features
                                        )

        #if self._img_transform:
        img_transformed = augxrdata.random_multime_transform(
            augfun=self._aug_transforms)

        self.trimage =  copy.deepcopy(img_transformed)

    def __init__(self, 
                 path,
                 fn_list,
                 target_Values,
                 shuffle = True,
                 scaler_target_values = None,
                 scaler_image_values = None,
                 features = None,
                 times = None,
                 get_zprofiles = False,
                 batch_size=1,
                 heightvariable = 'z',
                 workers = 6,
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
        self._heightprofile = get_zprofiles
        self._aug_transforms = augmentation_options
        self._heightvarname = heightvariable
        self.heightdata = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._workers = workers
        self.on_epoch_end()


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.fn_list) / self.batch_size))

    def initrun(self,idx):
        #
        
        self._get_randomtransform(idx)

        if self._heightprofile:
            self._getheighttsdata()
        
        if self._channelslast:
            x = self.trimage.swapaxes(1,2).swapaxes(2,3)
        else:
            x = self.trimage
        
        if self.heightdata is not None:
            x = [x, self.heightdata]


        y = self._target[idx]

        if self._scaler_target is not None:
            y  = self.__normalize(y)

        return x, y


    def __data_generation(self, list_ids):
        #import concurrent.futures as cf
        cloud_thread= []
        if self._workers is not None:
                
            pool = multiprocessing.Pool()
            pool = multiprocessing.Pool(processes=self._workers)
            #inputs = [0,1,2,3,4]
            listdata = pool.map(self.initrun, list_ids)
            pool.close()
            pool.join()
        #listdata= []
        #with cf.ProcessPoolExecutor(max_workers=5) as executor:
        #    for idx in list_ids:
        #        cloud_thread.append({executor.submit(self.initrun,
        #                                             idx): idx})

        #for future in cloud_thread:
        #    for i in cf.as_completed(future):
        #        listdata.append(i.result())
        else:
            listdata = [self.initrun(idx) for idx in list_ids]

        xlist = []
        ylist = []
        for xx, yy in listdata:
            xlist.append(xx)
            ylist.append(yy)

        
        if self._heightprofile:
            ximages = np.array([values[0] for values in xlist])
            xprofiles = np.array([values[1] for values in xlist])
            xlist = [ximages,xprofiles]
        else:
            xlist = np.array(xlist)
        
        return xlist, np.array(ylist)


    def __getitem__(self,idx):
        
        indexes = self._list_ids[idx*self.batch_size:(idx+1)*self.batch_size]

        # Find list of IDs
        x, y = self.__data_generation(indexes)

        return x, y
    
    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            
            if i == self.__len__()-1:
                self.on_epoch_end()
    
    def __normalize(self, x):
        return (x-self._scaler_target[0])/self._scaler_target[1]        

    #shuffles the dataset at the end of each epoch
    def on_epoch_end(self):
        self._list_ids = list(range(len(self.fn_list)))
        
        if self.shuffle == True:
            np.random.shuffle(self._list_ids)"""