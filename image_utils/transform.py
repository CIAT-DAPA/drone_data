
import random
from image_utils.general_functions import image_rotation,image_zoom,randomly_displace,clahe_img
#from utils.image_processing import ImageData
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import copy

from functools import wraps


def perform(fun, *args):
    return fun(*args)

def perform_kwargs(fun, **kwargs):
    return fun(**kwargs)


class FolderWithImages(object):
    @property
    def files_in_folder(self):
        
        return self._look_for_images()
    
    def __init__(self, path,suffix = '.jpg', shuffle = False, seed = None) -> None:
        
        self.path = path
        self.imgs_suffix = suffix
        self.shuffle = shuffle
        self.seed = seed
    
    def _look_for_images(self):
        filesinfolder = [i for i in os.listdir(self.path) if i.endswith(self.imgs_suffix)]
        if len(filesinfolder)==0:
            raise ValueError(f'there are not images in this {self.path} folder')

        if self.seed is not None and self.shuffle:
            random.seed(self.seed)
            random.shuffle(filesinfolder)

        elif self.shuffle:
            random.shuffle(filesinfolder)

        return filesinfolder


def summarise_trasstring(values):
    
    if type (values) ==  list:
        paramsnames = '_'.join([str(j) 
        for j in values])
    else:
        paramsnames = values

    return '{}'.format(
            paramsnames
        )

def _generate_shiftparams(img, max_displacement = None, xshift = None,yshift = None):

    if max_displacement is None:
        max_displacement = random.randint(5,20)/100
    
    if xshift is None:
        xoptions = list(range(-int(img.shape[0]*max_displacement),int(img.shape[0]*maxshift)))
        xshift = random.choice(xoptions)

    if yshift is None:
        yoptions = list(range(-int(img.shape[1]*max_displacement),int(img.shape[1]*maxshift)))
        yshift = random.choice(yoptions)

    return max_displacement, xshift, yshift

class ImageAugmentation(object):

    @property
    def _run_default_transforms(self):
        
        
        return  {
                'rotation': self.rotate_image,
                'zoom': self.expand_image,
                'clahe': self.clahe,
                'shift': self.shift_ndimage,
                'multitr': self.multi_transform
            }
        


    @property
    def _augmented_images(self):
        return self._new_images

    def multi_transform(self, img = None, 
                        chain_transform = ['zoom','shift','rotation'],
                         params = None, update = True):

        if img is None:
            img = self.img_data

        imgtr = copy.deepcopy(img)
        augmentedsuffix = {}
        for i in chain_transform:
            if params is None:
                imgtr = perform_kwargs(self._run_default_transforms[i],
                     img = imgtr,
                     update = False)
            else:
                imgtr = perform(self._run_default_transforms[i],
                     imgtr,
                     params[i], False)

            augmentedsuffix[i] = self._transformparameters[i]
        
        self._transformparameters['multitr'] = augmentedsuffix
        if update:
            self.updated_paramaters(tr_type = 'multitr')
            self._new_images['multitr'] = imgtr

        return imgtr

    def updated_paramaters(self, tr_type):
        self.tr_paramaters.update({tr_type : self._transformparameters[tr_type]})
        

    def rotate_image(self, img = None, angle = None, update = True):

        if img is None:
            img = copy.deepcopy(self.img_data)
        if angle is None:
            angle = random.randint(10,350)

        
        imgtr = image_rotation(img,angle = angle)
        self._transformparameters['rotation'] = angle
        if update:
            self.updated_paramaters(tr_type = 'rotation')
            self._new_images['rotation'] = imgtr

        return imgtr

    def expand_image(self, img = None, ratio = None, update = True):
        if ratio is None:
            ratio = random.choice([random.randint(-90,-60),random.randint(70,90)])
        if img is None:
            img = copy.deepcopy(self.img_data)
            
        imgtr = image_zoom(img, zoom_factor=ratio)
        self._transformparameters['zoom'] = ratio
        if update:
            self.updated_paramaters(tr_type = 'zoom')
            self._new_images['zoom'] = imgtr

        return imgtr
    

    def shift_ndimage(self,img = None, shift = None, update = True,
                      max_displacement = None):

        if max_displacement is None:
            max_displacement = random.randint(5,20)/100
        if img is None:
            img = copy.deepcopy(self.img_data)

        if shift is not None:
            xshift, yshift= shift   
        else:
            xshift, yshift = None, None

        imgtr, displacement =  randomly_displace(img, 
                                                 maxshift = max_displacement, 
                                                 xshift = xshift, yshift = yshift)

        self._transformparameters['shift'] = displacement
        if update:
            self.updated_paramaters(tr_type = 'shift')
            self._new_images['shift'] = imgtr#.astype(np.uint8)

        return imgtr#.astype(np.uint8)
    
    def clahe(self, img= None, thr_constrast = None, update = True):

        if thr_constrast is None:
            thr_constrast = random.randint(0,30)/10
        
        if img is None:
            img = copy.deepcopy(self.img_data)

        imgtr,_ = clahe_img(img, clip_limit=thr_constrast)

        self._transformparameters['clahe'] = thr_constrast
        if update:
            self.updated_paramaters(tr_type = 'clahe')
            self._new_images['clahe'] = imgtr
        return imgtr

    def random_augmented_image(self,img= None, update = True):
        if img is None:
            img = copy.deepcopy(self.img_data)
        
        imgtr = copy.deepcopy(img)
        augfun = random.choice(list(self._run_default_transforms.keys()))
        
        imgtr = perform_kwargs(self._run_default_transforms[augfun],
                     img = imgtr,
                     update = update)

        return imgtr

    def _transform_as_ids(self, tr_type):

        if type (self.tr_paramaters[tr_type]) ==  dict:
            paramsnames= ''
            for j in list(self.tr_paramaters[tr_type].keys()):
                paramsnames += 'ty_{}_{}'.format(
                    j,
                    summarise_trasstring(self.tr_paramaters[tr_type][j]) 
                )

        else:
            paramsnames = summarise_trasstring(self.tr_paramaters[tr_type])

        return '{}_{}'.format(
                tr_type,
                paramsnames
            )
    
    def augmented_names(self):
        transformtype = list(self.tr_paramaters.keys())
        augmentedsuffix = {}
        for i in transformtype:
            
            augmentedsuffix[i] = self._transform_as_ids(i)

        return augmentedsuffix

    def __init__(self, img, **kwargs) -> None:

        self.img_data = None
        if isinstance(img, str):
            self.img_data = cv2.imread(img)
        else:
            self.img_data = copy.deepcopy(img)

        self._transformparameters = {}
        self._new_images = {}
        self.tr_paramaters = {}

class ImageData(ImageAugmentation):
    

    @property
    def images_names(self):
        imgnames = {'raw': self.orig_imgname}
        augementednames = self.augmented_names()
        if len(list(augementednames.keys()))> 0:
            for datatype in list(augementednames.keys()):
                currentdata = {datatype: '{}_{}'.format(
                    imgnames['raw'],
                    augementednames[datatype])}
                imgnames.update(currentdata)

        return imgnames

    @property
    def imgs_data(self):
        imgdata = {'raw': self.img_data}
        augementedimgs = self._augmented_images
        if len(list(augementedimgs.keys()))> 0:
            for datatype in list(augementedimgs.keys()):
                currentdata = {datatype: augementedimgs[datatype]}
                imgdata.update(currentdata)

        return imgdata
        
    def __init__(self, path, img_id = None, **kwargs) -> None:
        
        if img_id is not None:
            self.orig_imgname = img_id
        else:
            self.orig_imgname = "image"

        try:
            img = cv2.imread(path)
            super().__init__(img, **kwargs)
        
        except:
            raise ValueError('check image filename')
    


class MultiChannelImage(ImageAugmentation):



    @property
    def imgs_data(self):
        imgdata = {'raw': self.mlchannel_data}
        augementedimgs = self._augmented_images
        if len(list(augementedimgs.keys()))> 0:
            for datatype in list(augementedimgs.keys()):
                currentdata = {datatype: augementedimgs[datatype]}
                imgdata.update(currentdata)

        return imgdata
    
    @property
    def images_names(self):
        imgnames = {'raw': self.orig_imgname}
        augementednames = self.augmented_names()
        if len(list(augementednames.keys()))> 0:
            for datatype in list(augementednames.keys()):
                currentdata = {datatype: '{}_{}'.format(
                    imgnames['raw'],
                    augementednames[datatype])}
                imgnames.update(currentdata)

        return imgnames
    
    def _tranform_channelimg_function(self, img, tr_name):

        if tr_name == 'multitr':
            params = self.tr_paramaters[tr_name]
            image = self.multi_transform(img=img,
                                chain_transform = list(params.keys()),params= params, update= False)

        else:
            
            image =  perform(self._run_default_transforms[tr_name],
                     img,
                     self.tr_paramaters[tr_name], False)

        return image

    def _transform_multichannel(self, img=None, tranformid = None,  **kwargs):
        
        newimgs = {}
        if img is not None:
            trimgs = img
        else:
            trimgs = self.mlchannel_data

        imgs= [perform_kwargs(self._run_default_transforms[tranformid],
                     img = trimgs[0],
                     **kwargs)]
         
        for i in range(1,trimgs.shape[0]):
            r = self._tranform_channelimg_function(trimgs[i],tranformid)
            imgs.append(r)

        imgs = np.stack(imgs, axis = 0)
        self._new_images[tranformid] = imgs
        
        return imgs

    def shift_multiimages(self, img=None, shift=None, max_displacement=None,update=True):

        self._new_images['shift'] =  self._transform_multichannel(img=img, 
                    tranformid = 'shift', shift = shift, max_displacement=max_displacement,update=update)
        return self._new_images['shift']

    def rotate_multiimages(self, img=None, angle=None, update=True):
        self._new_images['rotation'] = self._transform_multichannel(img=img, 
                    tranformid = 'rotation', angle = angle, update=update)
        
        return self._new_images['rotation']

    
    def expand_multiimages(self, img=None, ratio=None, update=True):

        self._new_images['zoom'] = self._transform_multichannel(img=img, 
                    tranformid = 'zoom', ratio = ratio, update=update)
        
        return self._new_images['zoom']


    def multitr_multiimages(self, img=None, 
                        chain_transform=['zoom','shift','rotation'], 
                        params=None, 
                        update=True):
        
        self._new_images['multitr'] = self._transform_multichannel(
                    img=img, 
                    tranformid = 'multitr', 
                    chain_transform = chain_transform,
                    params=params, update=update)
        
        return self._new_images['multitr']


    def __init__(self, img, img_id = None, formatorder = 'channels_first', **kwargs) -> None:

        if img_id is not None:
            self.orig_imgname = img_id
        else:
            self.orig_imgname = "image"

        if formatorder == 'channels_first':
            self._initimg = img[0]
        else:
            self._initimg = img[:,:,0]
        
        self.mlchannel_data = img

        super().__init__(self._initimg, **kwargs)
    


class Augmentation_Xrdata(MultiChannelImage):

    @property
    def _run_multichannel_transforms(self):
        return  {
                'rotation': self.rotate_multiimages,
                'zoom': self.expand_multiimages,
                #'clahe': self.clahe,
                'shift': self.shift_multiimages,
                'multitr': self.multitr_multiimages
            }

    @property
    def _run_random_choice(self):
        return  {
                'rotation': self.rotate_tempimages,
                'zoom': self.expand_tempimages,
                'shift': self.shift_multi_tempimages,
                'multitr': self.multtr_tempimages
            }

    def _multi_timetransform(self, tranformn,  **kwargs):
        imgs= [perform_kwargs(self._run_multichannel_transforms[tranformn],
                     img = self._initdate,
                     **kwargs)]
        
        for i in range(1,self.npdata.shape[1]):

            if tranformn != 'multitr':
                
                r = perform(self._run_multichannel_transforms[tranformn],
                               self.npdata[:,i,:,:],
                               self.tr_paramaters[tranformn],
                               False,
                               )
            else:
                r = perform_kwargs(self._run_multichannel_transforms[tranformn],
                               img=self.npdata[:,i,:,:],
                               params = self.tr_paramaters[tranformn],
                               update = False,
                               )
            imgs.append(r)

        imgs = np.stack(imgs, axis = 0)
        self._new_images[tranformn] = imgs

        return imgs

    def shift_multi_tempimages(self, shift=None, max_displacement=None):
        return self._multi_timetransform(tranformn = 'shift',
                                        shift= shift, 
                                        max_displacement= max_displacement)

    def rotate_tempimages(self, angle=None):
        return self._multi_timetransform(tranformn = 'rotation', angle = angle)

    def expand_tempimages(self, ratio=None):
        return self._multi_timetransform(tranformn = 'zoom', ratio = ratio)

    def rotate_tempimages(self, angle=None):
        return self._multi_timetransform(tranformn = 'rotation', angle = angle)

    def multtr_tempimages(self, img=None, chain_transform=['zoom', 'shift', 'rotation'], params=None):
        return self._multi_timetransform(tranformn = 'multitr', 
                                         chain_transform= chain_transform, 
                                         params = params)

    
    def random_multime_transform(self, augfun = None):
        availableoptions = list(self._run_multichannel_transforms.keys())+['raw']
        if augfun is None:
            augfun = random.choice(availableoptions)
        elif type(augfun) is list:
            augfun = random.choice(augfun)
        
        if augfun not in availableoptions:
            print(f"""that augmentation option is not into default parameters {availableoptions},
                     no transform was applied""")
            augfun = 'raw'
        
        if augfun == 'raw':
            imgtr = self.npdata.swapaxes(0,1)
        else:
            imgtr = perform_kwargs(self._run_random_choice[augfun])

        return imgtr

    def __init__(self, 
                 xrdata, 
                 img_id=None, 
                 formatorder='CDHW', 
                 variables = None, 
                 times = None, 
                 **kwargs) -> None:


        if variables is not None:
            self.npdata = copy.deepcopy(xrdata[variables].to_array().values)
            self.variables = variables
        else:
            self.npdata = copy.deepcopy(xrdata.to_array().values)
            self.variables = list(xrdata.keys())

        if formatorder == 'CDHW':
            channels_position = 'channels_first'
            self._initdate = copy.deepcopy(self.npdata[:,0,:,:])
        

        super().__init__(self._initdate, img_id, formatorder = channels_position, **kwargs)
