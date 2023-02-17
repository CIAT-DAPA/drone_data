from .general_functions import list_files
import numpy as np
from math import cos, sin, radians
import os

def expanded_yolobb(yolobb, origsize,expandsize):

    left, bottom = ((expandsize[1] - origsize[1])/2),((expandsize[0]-origsize[0])/2)

    label, x, y, w, h = np.array(yolobb).astype(np.float)

    xp = (int(x*origsize[0]) + left) /expandsize[0]
    yp = (int(y*origsize[0]) + bottom) /expandsize[1]

    wp = (w * origsize[1])/expandsize[1]
    hp = (h * origsize[0])/expandsize[0]

    return label, xp, yp, wp, hp


def calculate_expanded_label(yolobb,imageshape, ratio = 25):

    #st,_ = expand_npimage(image, ratio, keep_size=False)

    xnewsize = int(imageshape[0]*((float(ratio)/100.0*2.0)+1.0))
    ynewsize = int(imageshape[1]*((float(ratio)/100.0*2.0)+1.0))
    expanedbb = []
    for yolobbsingle in yolobb:
        expanedbb.append(expanded_yolobb(yolobbsingle, 
                            (imageshape[0],imageshape[1]),
                            (xnewsize,ynewsize)))

    return expanedbb

def rotate_xyxoords(x, y, anglerad, imgsize, xypercentage=True):
    center_x = imgsize[1] / 2
    center_y = imgsize[0] / 2

    xp = ((x - center_x) * cos(anglerad) - (y - center_y) * sin(anglerad) + center_x)
    yp = ((x - center_x) * sin(anglerad) + (y - center_y) * cos(anglerad) + center_y)

    if imgsize[0] != 0:
        if xp > imgsize[1]:
            xp = imgsize[1]
        if yp > imgsize[0]:
            yp = imgsize[0]

    if xypercentage:
        xp, yp = xp / imgsize[1], yp / imgsize[0]

    return xp, yp

def rotate_yolobb(yolobb,imageshape, angle):
    angclock = -1 * angle
    
    xc = float(yolobb[1]) * imageshape[1]
    yc = float(yolobb[2]) * imageshape[0]
    xr, yr = rotate_xyxoords(xc, yc, radians(angclock), imageshape)
    w_orig = yolobb[3]
    h_orig = yolobb[4]
    wr = np.abs(sin(radians(angclock))) * h_orig + np.abs(cos(radians(angclock)) * w_orig)
    hr = np.abs(cos(radians(angclock))) * h_orig + np.abs(sin(radians(angclock)) * w_orig)

    # l, r, t, b = from_yolo_toxy(origimgbb, (imgorig.shape[1],imgorig.shape[0]))
    # coords1 = rotate_xyxoords(l,b,radians(angclock),rotatedimg.shape)
    # coords2 = rotate_xyxoords(r,b,radians(angclock),rotatedimg.shape)
    # coords3 = rotate_xyxoords(l,b,radians(angclock),rotatedimg.shape)
    # coords4 = rotate_xyxoords(l,t,radians(angclock),rotatedimg.shape)
    # w = math.sqrt(math.pow((coords1[0] - coords2[0]),2)+math.pow((coords1[1] - coords2[1]),2))
    # h = math.sqrt(math.pow((coords3[0] - coords4[0]),2)+math.pow((coords3[1] - coords4[1]),2))
    return [yolobb[0], xr, yr, wr, hr]




def label_transform(imageshape, yolobb, augtype, combination, nrep = 1):
    
    if augtype == 'expand':
        attrs = float(combination[0])
        newbb = calculate_expanded_label( yolobb, imageshape,ratio = attrs)

    if augtype =='clahe_img':
        newbb = yolobb

    if augtype == 'hsv':
        newbb = yolobb
        
    if augtype == 'contrast':
        newbb = yolobb
    
    if augtype == 'blur':
        newbb = yolobb
    
    if augtype == 'rotate':
        attrs = float(combination[0])
        newbb = []
        for yolobbsingle in yolobb:
            newbb.append(rotate_yolobb(yolobbsingle, imageshape,angle = attrs))

    if nrep>1:
        newbb = [newbb for i in range(nrep)]
        
    return newbb


def save_yololabels(bbyolo, fn,outputdir = None):

    if outputdir is not None:
        fn = os.path.join(outputdir, fn)
    if bbyolo is not None:
        with open(fn, 'w') as dst:
            for i in range(len(bbyolo)):
                strlist = [str(int(bbyolo[i][0]))]
                for j in range(1,len(bbyolo[i])):
                    strlist.append(str(bbyolo[i][j]))
                if len(bbyolo)-1 == i:
                    dst.writelines(" ".join(strlist))
                else:
                    dst.writelines(" ".join(strlist) + '\n')


def from_yolo_toxy(yolo_style, size):
    dh, dw = size
    _, x, y, w, h = yolo_style

    l = int((x - w / 2) * dw)
    r = int((x + w / 2) * dw)
    t = int((y - h / 2) * dh)
    b = int((y + h / 2) * dh)

    if l < 0:
        l = 0
    if r > dw - 1:
        r = dw - 1
    if t < 0:
        t = 0
    if b > dh - 1:
        b = dh - 1

    return (l, r, t, b)


class LabelData:

    def __init__(self,
                 img_class,
                 label_type="yolo",
                 pattern=None):

        self.labeled_data = None

        if label_type == "yolo":

            source = img_class._input_path

            pattern = 'txt'
            self.labels_path_files = list_files(source, pattern=pattern)

            imgsfilepaths = img_class.jpg_path_files.copy()

            fn = [labelfn.split('\\')[-1][:-4] for labelfn in self.labels_path_files]
            fnorig = [labelfn for labelfn in self.labels_path_files]

            organized_labels = []
            labels_data = []
            idlist = []
            for i, imgfn in enumerate(imgsfilepaths):
                if '\\' in imgfn:
                    imgfn = imgfn.split('\\')[-1]
                if imgfn.endswith('jpg'):
                    imgfn = imgfn[:-4]

                lines = None
                datatxt = None
                if imgfn in fn:
                    datatxt = fnorig[fn.index(imgfn)]
                    with open(datatxt, 'rb') as src:
                        lines = src.readlines()
                    linesst = [z.decode().split(' ') for z in lines]
                    idlist.append(i)
                    lines = []
                    for z in range(len(linesst)):
                        ls = [int(linesst[z][0])]
                        for i in range(1,len(linesst[z])):
                            ls.append(float(linesst[z][i]))
                        lines.append(ls)

                organized_labels.append(datatxt)
                labels_data.append(lines)

            # self.labeled_data = {'images': [img_class._augmented_data['original']['imgs'][i] for i in idlist],
            #                     'yolo_boundary':labels_data,
            #                     'filenames':organized_labels}    
            
            self.labels = labels_data
            self._path = organized_labels