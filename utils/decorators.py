import os
import cv2

def check_output_fn(func):
    def inner(file, path, fn, suffix):
        try:
            if not os.path.exists(path):
                os.mkdir(path)
        except:
            raise ValueError('the path can not be used')
        if not fn.endswith(suffix):
            fn = os.path.join(path, fn+suffix)
        else:
            fn = os.path.join(path, fn)
            
        return func(file, path=path, fn=fn)
    
    return inner

def check_image_size(func):
    def inner(image, outputsize):
        swapxes = False
        if image.shape[-1] != 3:
            image= image.swapaxes(0,1).swapaxes(2,1)
            swapxes = True
            
        if image.shape[0] != outputsize[0] and image.shape[1] != outputsize[1]:
            image = cv2.resize(image, outputsize)

        if swapxes:
            image = image.swapaxes(2,1).swapaxes(0,1)
            
        return func(image, outputsize)
    
    return inner
