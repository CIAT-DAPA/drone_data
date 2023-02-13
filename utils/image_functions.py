import scipy.signal
import numpy as np
import random
from scipy import ndimage
from skimage.transform import SimilarityTransform
from skimage.transform import warp

from PIL import Image, ImageOps
from math import cos, sin, radians
from skimage.draw import line_aa
from scipy.spatial import ConvexHull
import warnings
import cv2 as cv


# https://en.wikipedia.org/wiki/Histogram_equalization
def hist_equalization(np2dimg):
    if 'f' in np2dimg.dtype.str:
        np2dimg[np.isnan(np2dimg)] = 0
        np2dimg = np2dimg.astype('uint8')

    hist,_ = np.histogram(np2dimg.flatten(),256,[0,256])
    cdf = hist.cumsum()
    #cdf_normalized = cdf * float(hist.max()) / cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    return cdf[np2dimg]
    
def hist_3dimg(np3dimg):
    ref = np.min(np3dimg.shape)
    mind = [i for i in range(len(np3dimg.shape)) if np3dimg.shape[i] == ref]
    imgeqlist =[]
    for i in range(np3dimg.shape[mind[0]]):
        
        imgeqlist.append(hist_equalization(np3dimg[i]))
    
    return np.array(imgeqlist)

def change_bordersvaluesasna(nptwodarray, bufferna):
    intborderx = int(nptwodarray.shape[0]*(bufferna/100))
    intbordery = int(nptwodarray.shape[1]*(bufferna/100))
    nptwodarray[
        (nptwodarray.shape[0]-intborderx):nptwodarray.shape[0],
            :] = np.nan
    nptwodarray[:,
            (nptwodarray.shape[1]-intbordery):nptwodarray.shape[1]] = np.nan
    nptwodarray[:,
            0:intbordery] = np.nan
    nptwodarray[0:intborderx,:] = np.nan

    return nptwodarray

def getcenter_from_hull(npgrayimage, buffernaprc = 15):
    nonantmpimg = npgrayimage.copy()
    if buffernaprc is not None:
        nonantmpimg = change_bordersvaluesasna(nonantmpimg, bufferna=buffernaprc)

    nonantmpimg[np.isnan(nonantmpimg)] = 0

    coords = np.transpose(np.nonzero(nonantmpimg))
    hull = ConvexHull(coords)

    cx = np.mean(hull.points[hull.vertices,0])
    cy = np.mean(hull.points[hull.vertices,1])

    return int(cx),int(cy)


def border_distance_fromgrayimg(grimg):
    contours, _ = cv.findContours(grimg, 
                                  cv.RETR_TREE,
                                  cv.CHAIN_APPROX_SIMPLE)
                                  
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly = cv.approxPolyDP(c, 3, True)
        centers[i], radius[i] = cv.minEnclosingCircle(contours_poly)

    centers = centers[np.where(radius == np.max(radius))[0][0]]
    radius = radius[np.where(radius == np.max(radius))[0][0]]

    return centers,radius

def cross_image(im1, im2):
    # get rid of the color channels by performing a grayscale transform
    # the type cast into 'float' is to avoid overflows
    im1_gray = np.sum(im1.astype('float'), axis=2)
    im2_gray = np.sum(im2.astype('float'), axis=2)

    # get rid of the averages, otherwise the results are not good
    im1_gray -= np.mean(im1_gray)
    im2_gray -= np.mean(im2_gray)

    # calculate the correlation image; note the flipping of onw of the images
    return scipy.signal.fftconvolve(im1_gray, im2_gray[::-1, ::-1], mode='same')


def phase_convolution(refdata, targetdata):
    corr_img = cross_image(refdata,
                           targetdata)
    shape = corr_img.shape

    midpoints = np.array([np.fix(axis_size / 2) for axis_size in shape])
    maxima = np.unravel_index(np.argmax(corr_img), shape)
    shifts = np.array(maxima, dtype=np.float64)

    shifts = np.array(shifts) - midpoints

    return shifts


def register_image_shift(data, shift):
    tform = SimilarityTransform(translation=(shift[1], shift[0]))
    imglist = []
    for i in range(data.shape[2]):
        imglist.append(warp(data[:, :, i], inverse_map=tform, order=0, preserve_range=True))

    return np.dstack(imglist)


def cartimg_topolar_transform(nparray, anglestep = 5, max_angle = 360, expand_ratio = 40, nathreshhold = 5):

    xsize = nparray.shape[1]
    ysize = nparray.shape[0]
    
    if expand_ratio is None:
        mayoraxisref = [xsize,ysize] if xsize >= ysize else [ysize,xsize]
        expand_ratio = (mayoraxisref[0]/mayoraxisref[1] - 1)*100

    newwidth = int(xsize * expand_ratio / 100)
    newheight = int(ysize * expand_ratio / 100)

    # exand the image for not having problem whn one axis is bigger than other
    pil_imgeexpand = ImageOps.expand(Image.fromarray(nparray), 
                                     border=(newwidth, newheight), fill=np.nan)

    
    listacrossvalues = []
    distances = []
    # the image will be rotated, then the vertical values were be extracted with each new angle
    for angle in range(0, max_angle, anglestep):
        
        imgrotated = pil_imgeexpand.copy().rotate(angle)
        imgarray = np.array(imgrotated)
        cpointy = int(imgarray.shape[0]/2)
        cpointx = int(imgarray.shape[1]/2)
        if np.isnan(imgarray[cpointy,cpointx]):
            cpointy, cpointx = getcenter_from_hull(imgarray)

        valuesacrossy = []
        
        i=(cpointy+0)
        coordsacrossy = []
        # it is important to have non values as nan, if there are to many non values in a row it will stop
        countna = 0 
        while (countna<nathreshhold) and (i<(imgarray.shape[0]-1)):
            
            if np.isnan(imgarray[i,cpointx]):
                countna+=1
            else:
                coordsacrossy.append(i- cpointy)
                valuesacrossy.append(imgarray[i,cpointx])
                countna = 0
            i+=1

        distances.append(coordsacrossy)
        listacrossvalues.append(valuesacrossy)
    
    maxval = 0
    nrowid =0 
    for i in range(len(distances)):
        if maxval < len(distances[i]):
            maxval = len(distances[i])
            
            nrowid = distances[i][len(distances[i])-1]
    
    for i in range(len(listacrossvalues)):
        listacrossvalues[i] = listacrossvalues[i] + [np.nan for j in range((nrowid+1) - len(listacrossvalues[i]))]
    

    return [distances, np.array(listacrossvalues)]


def radial_filter(nparray, anglestep = 5, max_angle = 360, nathreshhold = 5):
    

    
    center_x = int(nparray.shape[1]/2)
    center_y = int(nparray.shape[0]/2)

    origimg = nparray.copy()
    if np.isnan(origimg[center_y,center_x]):
        center_y, center_x = getcenter_from_hull(origimg)

    if np.isnan(origimg[center_y,center_x]):
        warnings.warn("there are no data in the image center")
        origimg[center_y,center_x] = 0

    modimg = np.empty(nparray.shape)
    modimg[:] = np.nan

    #mxanglerad = radians(360)
    for angle in range(0,max_angle,anglestep):

        anglerad = radians(angle)
        xp = ((center_x*2 - center_x) * cos(anglerad) - (center_y*2 - center_y) * sin(anglerad) + center_x)
        yp = ((center_x*2 - center_x) * sin(anglerad) + (center_y*2 - center_y) * cos(anglerad) + center_y)

        x1, y1 = center_x, center_y
        x2, y2 = xp,yp
        rr, cc, _ = line_aa(y1, x1, int(y2),int(x2))
        countna = 0 
        for i,j in zip(rr,cc):
            #print(i,j,m)
            if i < (nparray.shape[0]) and j < (nparray.shape[1]) and j >= 0 and i >=0:
                         
                if countna>=nathreshhold:
                    #modimg[i,j] = np.nan
                    break
                try:
                    if np.isnan(origimg[i,j]):
                        countna+=1
                    else:
                        modimg[i,j] = origimg[i,j]
                        countna = 0
                except:
                    break
            else:
                break
    
    return modimg



## data augmentation

#https://keras.io/examples/vision/3D_image_classification/

#https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions

import cv2

def cv2_clipped_zoom(img, zoom_factor=0):

    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    ------
    Args:
        img : ndarray
            Image array
        zoom_factor : float
            amount of zoom as a ratio [0 to Inf). Default 0.
    ------
    Returns:
        result: ndarray
           numpy ndarray of the same shape of the input img zoomed by the specified factor.          
    """
    if zoom_factor == 0:
        return img


    height, width = img.shape[:2] # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
    
    ### Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1,x1,y2,x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_img = img[y1:y2, x1:x2]
    
    # Handle padding when downscaling
    resize_height, resize_width = min(new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (height - resize_height) // 2, (width - resize_width) //2
    pad_height2, pad_width2 = (height - resize_height) - pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1, pad_width2)] + [(0,0)] * (img.ndim - 2)

    result = cv2.resize(cropped_img, (resize_width, resize_height), 
                        interpolation= cv2.INTER_LINEAR
                        )
    result = np.pad(result, pad_spec)
    assert result.shape[0] == height and result.shape[1] == width
    result[result<0.000001] = 0.0
    return result


def scipy_rotate(volume,angles = [-330,-225,-180,-90, -45, -15, 15, 45, 90,180,225, 330]):
        # define some rotation angles
        
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

def clipped_zoom(img, zoom_factor, **kwargs): 
    h, w = img.shape[:2] # For multichannel images we don't want to apply the zoom factor to the RGB # dimension, so instead we create a tuple of zoom factors, one per array 
    # dimension, with 1's for any trailing dimensions after the width and height. 
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2) # Zooming out 
    if zoom_factor < 1: # Bounding box of the zoomed-out image within the output array 
        zh = int(np.round(h * zoom_factor)) 
        zw = int(np.round(w * zoom_factor)) 
        top = (h - zh) // 2 
        left = (w - zw) // 2 # Zero-padding 
        out = np.zeros_like(img) 
        out[top:top+zh, left:left+zw] = ndimage.zoom(img, zoom_tuple, **kwargs) # Zooming in 
    elif zoom_factor > 1: # Bounding box of the zoomed-in region within the input array 
        zh = int(np.round(h / zoom_factor)) 
        zw = int(np.round(w / zoom_factor)) 
        top = (h - zh) // 2 
        left = (w - zw) // 2 
        out = ndimage.zoom(img[top:top+zh, left:left+zw], zoom_tuple, **kwargs) # `out` might still be slightly larger than `img` due to rounding, so
         # trim off any extra pixels at the edges 
        
        trim_top = ((out.shape[0] - h) // 2) 
        trim_left = ((out.shape[1] - w) // 2) 
        out = out[trim_top:trim_top+h, trim_left:trim_left+w] # 
    else: 
        out = img 
    
    return out 

def randomly_zoom(volume):
        # define some rotation angles
        zooms = [1.75,1.5,1.25, 0.75, 0.85]
        # pick angles at random
        z = random.choice(zooms)
        
        # rotate volume
        volume = clipped_zoom(volume, z)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

def randomly_cv2_zoom(mltimage):
        # define some rotation angles
        zooms = [1.5,1.25, 0.75, 0.85]
        # pick angles at random
        z = random.choice(zooms)
        
        # zoom image
        # HWDC
        stackedimgs = []
        if len(mltimage.shape)>3:
            for t in range(mltimage.shape[2]):
                stackedimgs.append(cv2_clipped_zoom(mltimage[:,:,t,:].copy(), z))
            stackedimgs = np.stack(stackedimgs, axis = 2)
        else:
            stackedimgs = cv2_clipped_zoom(mltimage.copy(), z)

        return stackedimgs

def randomly_shiftimage(img,maxshift = 0.15):
    xoptions = list(range(-int(img.shape[0]*maxshift),int(img.shape[0]*maxshift)))
    yoptions = list(range(-int(img.shape[1]*maxshift),int(img.shape[1]*maxshift)))
            # pick angles at random
    xshift = random.choice(xoptions)
    yshift = random.choice(yoptions)
    if len(img.shape)>3:
        stackedimgs = []
        for t in range(img.shape[2]):
            stackedimgs.append(register_image_shift(img[:,:,t,:].copy(), [xshift,yshift]))
        stackedimgs = np.stack(stackedimgs, axis = 2)
    else:
        stackedimgs = register_image_shift(img.copy(), [xshift,yshift])

    return stackedimgs



class DataAugmentation:

    def rotate_ndimage(self,ntimes =1,
                       angles = [-330,-225,-180,-90, -45, -15, 15, 45, 90,180,225, 330]):
        newdata = []
        for i in range(ntimes):
            newdata = scipy_rotate(self.image,angles = angles )
        
        return newdata

    def expand_ndimage(self,ntimes = 1):
        newdata = []

        for i in range(ntimes):
            newdata = randomly_cv2_zoom(self.image )
        
        return newdata
    
    def shift_ndimage(self,ntimes = 1,**kargs):
        newdata = []

        for i in range(ntimes):
            newdata = randomly_shiftimage(self.image, **kargs)
        
        return newdata
        
    def __init__(self,
    
                 nd_image,
                 ):
        
        self.image = nd_image


def tranform_z2dinto_profiles(z2dimage, axis = 'x',
                              zmin = 0, zmax = 60,cms = 1, 
                              scalez = 10, slices_step = 1, initval = 1, **kwargs
                             ):
    
    """
    a functions to reshape the [x y z] data to a [x z chanel] image

    ----------
    Parameters
    data : numpy 2d array
        an array that contains the chanel data to be transformed as x z axis
    zvalues: numpy 2d array
        an array that contains the z values
    axis :int, optional
        which axis will be transformed along with z [ 'x' , 'y' ]
    
    zmin : int, optional
        a value that will be taken as minimun for z axis
    zmax : int, optional
        a value that will be taken as maximun for z axis

    slices_step: int, optional
        a integer value to set
    ----------
    Returns
    numpy 3d array [chanel, x, z]
    """

def from_to2d_zarray(zvalues, 
                     fill_values = None,
                     axis = 'x', 
                     zmin = 0, zmax = 60,cms = 1, scalez = 10, slices_step = 1, initval = 1, **kwargs):

    """
    a functions to reshape the [x y z chanel] data to a [x z chanel] image

    ----------
    Parameters
    data : numpy 2d array
        an array that contains the chanel data to be transformed as x z axis
    zvalues: numpy 2d array
        an array that contains the z values
    axis :int, optional
        which axis will be transformed along with z [ 'x' , 'y' ]
    
    zmin : int, optional
        a value that will be taken as minimun for z axis
    zmax : int, optional
        a value that will be taken as maximun for z axis

    slices_step: int, optional
        a integer value to set
    ----------
    Returns
    numpy 3d array [chanel, x, z]
    """

    if axis=='x':
        lenaxis = zvalues.shape[0]
        slicedstr = 'zvalues[j]'
        if fill_values is not None:
            sentvalues = '[data[j]]'
        else:
            sentvalues = 'fill_values'
    else:
        lenaxis = zvalues.shape[1]
        slicedstr = 'zvalues[:,j]'
        if fill_values is not None:
            sentvalues = '[data[:,j]]'
        else:
            sentvalues = 'fill_values'


    zimg = np.array([i for i in range(zmin*scalez, zmax*scalez, int(float(cms) * float(scalez)))])
    

    slicelist = []
    for j in range(initval*slices_step,(lenaxis-(slices_step)),slices_step):
        slicedataz = eval(slicedstr)
        if np.sum(np.logical_not(np.isnan(slicedataz))) > 0:
            slicedata = eval(sentvalues)
            z2dimg = singlexy_to2d_zarray(
                                           
                                          slicedataz, 
                                          fill_Values = slicedata,
                                          scalez = scalez, 
                                          referencearray = zimg,**kwargs)[0]
        else:
            z2dimg = np.zeros((len(zimg),len(slicedataz)))
        slicelist.append(z2dimg)

    return np.array(slicelist)    


#@ray.remote
def transform_to_heightprofile(
                     z2img, 
                     fill_values = None,
                     slices_step = 6,
                     axis = 0, 
                     zmax_cm = 60,
                     scalez = 1, 
                     
                     nslices = None,
                     initial_slice = 1, **kwargs):

    """
    a functions to reshape the [x y z chanel] data to a [x z chanel] image

    ----------
    Parameters
    zvalues: numpy 2d array
        an array that contains the z values
    fill_values : numpy 2d array, optional
        an array that contains the chanel data to be transformed as x z axis1
    axis :int, optional
        which axis will be transformed along with z [ 0 , 1 ]
    
    scalez : int, optional
        a value that will be taken as minimun for scaling z
    zmax_cm : int, optional
        a value that will be taken as maximun for z axis in cm

    slices_step: int, optional
        a integer value to set
    ----------
    Returns
    numpy 3d array [chanel, x, z]
    """

    if axis == 0:
        datatotransform = z2img
    else:
        datatotransform = z2img.swapaxes(1,0)
        if fill_values is not None:
            fill_values = fill_values.swapaxes(1,0)

    lenaxis = datatotransform.shape[0]
    newheight = int(zmax_cm*scalez)
    if nslices is not None:
        slices_step = int(lenaxis/nslices)
        print(slices_step)
        
    slicelist = []
    for j in range(initial_slice*slices_step,
                    (lenaxis-(slices_step)),slices_step):
        slicedataz = datatotransform[j]
        if np.sum(np.logical_not(np.isnan(slicedataz))) > 0:
            z2dimg = singleline_to2d_slice(datatotransform[j], 
                                            fill_Values = datatotransform[j],
                                            scalez = scalez, 
                                            referenheight= newheight,
                                            **kwargs
                                            )

        else:
            z2dimg = np.zeros((newheight,len(slicedataz)))
        slicelist.append(z2dimg)

    return np.array(slicelist) 



def singleline_to2d_slice(zvalues, fill_Values = None, scalez = 1, 
                         referenheight = None, 
                         barstyle = True, flip =True, fliplefttoright = False):

    """
    a functions to reshape the [x y z chanel] data to a [x z chanel] image

    ----------
    Parameters
    fill_Values : numpy 2d array, optional
        an array that contains the chanel data to be transformed as x z axis
    zvalues: numpy 2d array
        an array that contains the z values

    scalez: floar, optional
        a numerical value that will be used to scale height images
    
    ----------
    Returns
    numpy 3d array [chanel, x, z]
    """

    if fill_Values is None:
        barstyle = False
    else:
        barstyle = True
        
    npzeros = np.zeros((len(zvalues),referenheight))
    for xi in range(len(zvalues)):
        xivals = zvalues[xi]
        if not np.isnan(zvalues[xi]):
            zpos = (np.round(xivals,1)*scalez).astype(int)
            zpos = zpos - zpos%scalez
            
            if barstyle:
                npzeros[xi,:int((zpos))] = fill_Values[xi]
            else:
                npzeros[xi,:int((zpos))] = 1

    if flip:
        npzeros = npzeros.swapaxes(0,1)
        npzeros = Image.fromarray(npzeros)
        npzeros = np.array(ImageOps.flip(npzeros))
    if fliplefttoright:
        npzeros = np.array(Image.fromarray(npzeros).transpose(Image.FLIP_LEFT_RIGHT))
    
    return npzeros


# create filters by contour

def remove_smallpixels(img,imgrefpos = 0,thresh = 0.7):
    channelimg = img[imgrefpos].copy().astype(np.uint8)
    channelimg[channelimg<0] = 0
    channelimg[channelimg>0] = 255
    ## get mask
    smallpixelsmask = calculate_mask_contourarea(channelimg, threhold = thresh)
    
    newimgs = np.zeros(img.shape, dtype=img.dtype)
    
    for i in range(img.shape[0]):
        newimgs[i] = cv2.bitwise_and(img[i], img[i], mask=smallpixelsmask)
    
    return newimgs

def calculate_mask_contourarea(image,threhold = 0.7):
    ret, thresh = cv2.threshold(image, 127, 255, 0, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # create an empty mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    allarea = mask.shape[0] * mask.shape[0]
    cntmaxarea = np.max([cv2.contourArea(cnt) for cnt in contours])

    for i, cnt in enumerate(contours):
        #if hierarchy[0][i][2] != -1:
        if (cv2.contourArea(cnt)/allarea)/(cntmaxarea/allarea) >threhold:
            cv2.drawContours(mask, [cnt], 0, (255), -1)
        
    return mask
    