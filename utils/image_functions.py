import scipy.signal
import numpy as np

from skimage.transform import SimilarityTransform
from skimage.transform import warp

from PIL import Image, ImageOps
from math import cos, sin, radians
from skimage.draw import line_aa
from scipy.spatial import ConvexHull
import warnings
import cv2 as cv


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
    
    xborder = nparray.shape[1]
    yborder = nparray.shape[0]
    
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


