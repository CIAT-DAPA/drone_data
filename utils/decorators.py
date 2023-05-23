import os
    
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
