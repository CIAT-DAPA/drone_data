import os
import cv2

def check_output_fn(func):
    """
    A decorator that checks the existence of a specified path, creates it if necessary, 
    and constructs the full file path with the correct suffix.

    Parameters:
    func (function): A function that requires path, filename (fn), and suffix as arguments.

    Returns:
    function: A wrapper function that adds path validation and adjustment to the original function.

    Raises:
    ValueError: If the specified path cannot be used or created.
    """
    
    def inner(file, path, fn, suffix):
        try:
            if not os.path.exists(path):
                os.makedirs(path)
        except Exception as e:
            raise ValueError(f"Unable to use or create the specified path: {path}. Error: {e}")

        fn = os.path.join(path, fn if fn.endswith(suffix) else fn + suffix)
          
        return func(file, path=path, fn=fn)
    
    return inner

def check_image_size(func):
    
    """
    A decorator that ensures an image is of a specified size, resizing it if necessary. 
    Handles images with varying axis orders.

    Parameters:
    func (function): A function that requires image and outputsize as arguments.

    Returns:
    function: A wrapper function that adds image size validation and resizing to the original function.

    Notes:
    This decorator assumes the image is either in HWC (Height x Width x Channels) format 
    or CHW (Channels x Height x Width) format.
    """
    
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
