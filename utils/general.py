import math
import pandas as pd
import numpy as np

MSVEGETATION_INDEX = {# rgb bands
'grvi': '(green_ms - red_ms)/(green_ms + red_ms)',
'grvi_rgb': '(green - red)/((green + red)+0.00001)',
'mgrvi': '((green_ms*green_ms) - (red_ms*red_ms))/((green_ms*green_ms) + (red_ms*red_ms))',
'mgrvi_rgb': '((green*green) - (red*red))/(((green*green) + (red*red))+0.00001)',
'rgbvi': '((green_ms*green_ms) - (blue_ms*red_ms))/ ((green_ms*green_ms) + (blue_ms*red_ms))',
'rgbvi_rgb': '((green*green) - (blue*red))/ (((green*green) + (blue*red))+0.00001)',
 # nir indexes
 'ndvi': '(nir - red_ms)/(nir + red_ms)',
'ndre': '(nir - edge)/(nir + edge)',
'gndvi': '(nir - green_ms)/(nir + green_ms)',
'regnvi': '(edge - green_ms)/(edge + green_ms)',
'reci': '(nir / edge) - 1',
'negvi': '((nir*nir) - (edge*green_ms))/ ((nir*nir) + (edge*green_ms))',
'savi':  '((nir - red_ms) / (nir + red_ms + 0.5)) * (1.5)',
'greenness': 'green_ms / (red_ms + green_ms + blue_ms)',
'redness': 'red_ms / (red_ms + green_ms + blue_ms)',
'blueness': 'blue_ms / (red_ms + green_ms + blue_ms)',
'yellowness': '(green_ms - blue_ms)/(green_ms + blue_ms)',
'yi_edge': '-1*(green_ms - 2*red_ms + edge)',
#'yi': 'blue_ms - 2*green_ms + red_ms'

}


VEGETATION_INDEX = {# rgb bands
'grvi': '(green - red)/(green + red)',
'grvi_eq': '(green_eq - red_eq)/(green_eq + red_eq)',
'mgrvi': '((green*green) - (red*red))/((green*green) + (red*red))',
'rgbvi': '((green*green) - (blue*red))/ ((green*green) + (blue*red))',
 # nir indexes
'ndvi': '(nir - red)/(nir + red)',
'ndre': '(nir - edge)/(nir + edge)',
'gndvi': '(nir - green)/(nir + green)',
'regnvi': '(edge - green)/(edge + green)',
'reci': '(nir / edge) - 1',
'negvi': '((nir*nir) - (edge*green))/ ((nir*nir) + (edge*green))'}




def find_postinlist(listvalues, refvalue):
    """find the position of value in a list

    Args:
        listvalues (list): list of values in where the search is gonna be take
        refvalue (float): which value is gonna be look for in the list

    Returns:
        int: position in the list
    """
    listvalues.sort()
    posinlist = [i for i in range(len(listvalues)-1) if refvalue <= listvalues[i+1] and refvalue >= listvalues[i]]
    if len(posinlist)==0:
        posinlist = None
    else:
        posinlist = posinlist[0]
    return posinlist


def euclidean_distance(p1,p2):
    return math.sqrt(
        math.pow(p1[0] - p2[0],2) + math.pow(p1[1] - p2[1],2))


        
def organize_scaler(mscalervals, rgbscalervals, varnames, 
                    rgbchannels = ['blue','green','red'],
                    mschannels = ['blue_ms','green_ms','red_ms','edge','nir'],
                    standarizationtype = 'standarization'):
    orscaler = {}
    for channel in varnames:
        if channel in rgbchannels:
            orscaler[channel] = rgbscalervals[channel][standarizationtype]
        if channel in mschannels:
            orscaler[channel] = mscalervals[standarizationtype]
    
    return orscaler

def calculate_fd(pddata, xvalues):
    dx = np.diff(xvalues)
    maskwrongvalues = dx<100.
    dx = dx[maskwrongvalues]
    fd = pddata.apply(lambda y: np.diff(y)[maskwrongvalues]/dx, axis = 1)
    #fddf = pd.concat([i for i in fd], axis = 1)pd.concat([pd.DataFrame(list(i)).T for i in fd], axis = 1)
    return pd.concat([pd.DataFrame(list(i)).T for i in fd], axis = 0).reset_index().drop(['index'], axis = 1)

