



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
'savi':  '((nir - red_ms) / (nir + red_ms + 0.5)) * (1.5)'}


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



