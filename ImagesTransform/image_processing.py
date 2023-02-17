from dataclasses import dataclass
import os
import glob
import random
import itertools
from unittest import removeResult

#from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from PIL import Image

# import tensorflow as tf
import io
import numpy as np
from numpy.lib.function_base import kaiser
from six import BytesIO

#from utils.tf_model_detection import single_image_detection
from .plt_functions import plot_single_image, plot_single_image_odlabel
from .image_functions import blur_image, split_image, change_images_contrast, from_array_2_jpg
from .image_functions import rotate_npimage, expand_npimage, clahe_img, shift_hsv


from .general_functions import get_filename_frompath,list_files
from .bb_assignment import from_yolo_toxy, LabelData, label_transform, save_yololabels

from tqdm import tqdm



# TODO: create a different class for reading pascal data as images, it is not working
# 


def percentage_to_bb(bb, size):
    ymin = int(bb[0] * size[1])  # xmin
    xmin = int(bb[1] * size[0])  # ymin
    ymax = int(bb[2] * size[1])  # xmax
    xmax = int(bb[3] * size[0])  # ymax

    return np.array([[xmin, ymin, xmax, ymax]])


def bb_topercentage(bb, size):
    xmin = bb[0] / size[1]  # xmin
    ymin = bb[1] / size[0]  # ymin
    xmax = bb[2] / size[1]  # xmax
    ymax = bb[3] / size[0]  # ymax

    return np.array([[ymin, xmin, ymax, xmax]])




#def get_image_path_from_xml(data):
#    """

#    :param xml_file:
#    :return:
#    """

#    bs_data = BeautifulSoup(data, 'xml')
#    path_b4 = bs_data.find_all('path')
#    image_b4 = bs_data.find_all('filename')

#    return [path_b4[0].text,
#            image_b4[0].text]


def get_bbox(b4attribute):
    """

    :param b4attribute:
    :return: list
    """
    return [int(b4attribute.find_all('xmin')[0].text),
            int(b4attribute.find_all('ymin')[0].text),
            int(b4attribute.find_all('xmax')[0].text),
            int(b4attribute.find_all('ymax')[0].text)]

"""
def get_bounding_box_from_xml(data):

    This function only works with pascalIVOC annotation format
    more information in https://roboflow.com/formats/pascal-voc-xml
    :param xml_file:
    :return: list [xmin, ymin, xmax, ymax]

    bs_data = BeautifulSoup(data, 'xml')
    boundingbox_list = bs_data.find_all('bndbox')
    if len(boundingbox_list) > 1:
        bb = []
        for i in range(len(boundingbox_list)):
            bb.append(get_bbox(boundingbox_list[i]))
    else:
        bb = [get_bbox(boundingbox_list[0])]

    return bb
"""
"""
def get_image_orgsize_from_xml(data):


    bs_data = BeautifulSoup(data, 'xml')
    size_list = bs_data.find_all('size')

    return [int(size_list[0].find_all('height')[0].text),
            int(size_list[0].find_all('width')[0].text)]
"""

def load_image_into_numpy_array(path, scale_factor=None,
                                size=None):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
    path: a file path.

    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """

    # img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(io.open(path, "rb", buffering=0).read()))
    (im_width, im_height) = image.size
    if scale_factor is not None:
        im_width = int(im_width * scale_factor / 100)
        im_height = int(im_height * scale_factor / 100)
        image = image.resize((im_width, im_height))
    if size is not None:
        im_width = size[1]
        im_height = size[0]
        image = image.resize((im_width, im_height))
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def export_masked_image(image, ground_box, filename):
    bb_int = percentage_to_bb(ground_box['detection_boxes'][0],
                              (image.shape[1], image.shape[0]))

    image = Image.fromarray(image)
    image = image.crop(box=bb_int[0])

    image.save(filename)
    print("Image saved: {}".format(filename))


class ImageData:

    @property
    def od_labels(self):

        return self._labels
    
    @property
    def images_data(self):
        data = {}
        for datatype in list(self._augmented_data.keys()):
            currentdata = {datatype: self._augmented_data[datatype]['imgs']}

            data.update(currentdata)

        return data
    
    @property
    def images_names(self):
        imgnames = {}
        for datatype in list(self._augmented_data.keys()):
            currentdata = {datatype: self._augmented_data[datatype]['names']}

            imgnames.update(currentdata)

        return imgnames
        
    def _check_applyaug2specifictype(self, augtype = None, labelnewtype = None):

        if augtype is not None:
            augtype = [augtype] if type(augtype) != list else augtype
        else:
            augtype = self._augmented_data.keys()

        augtype = [i for i in augtype if i != labelnewtype]

        return augtype

    def _augmentation(self, func, datainput, label = None, samplesize=None,seed = 123, **kwargs):
        """
        samplesize: int, percentage of data which will be sampled
        """
        if label is None:
            label= "augmentation_{}".format(
                len(list(self._augmented_data.keys()))+1)
        listimgs = []
        fnlist = []
        odlabels = []
        for datatype in datainput:
            imgstoprocess = self._augmented_data[datatype]
            idimages = range(len(imgstoprocess['imgs']))
            if samplesize is not None:
                lensample = int(len(imgstoprocess['imgs'])*float(samplesize)/100.)
                random.seed(seed)
                idimages = random.sample(idimages, lensample)

            for idimage in idimages:
                
                newdata, combs = eval("{}(imgstoprocess['imgs'][idimage],**{})".format(
                                      func,
                                      kwargs))
                 
                fnlist.append([
                    "{}_{}_{}".format(
                        imgstoprocess['names'][idimage],
                        label,
                        comb) for comb in combs]) 

                if self.od_labels[datatype] is not None:

                    bb = eval(
                            "label_transform(imgstoprocess['imgs'][idimage].shape,self.od_labels[datatype][idimage],label,combs)")
                    if label == 'contrast':
                        for i in range(len(combs)):
                            odlabels.append(bb)
                    else:
                        odlabels.append(bb)
                            


                listimgs.append(newdata)
        if label in ['contrast','tiles']:
            listimgs= list(itertools.chain.from_iterable(listimgs))

        newdata = {label: {'imgs': listimgs,
                           'names': list(itertools.chain.from_iterable(fnlist))}}

        self._augmented_data.update(newdata)
        print('{} were added to images data'.format(len(listimgs)))

        # label
        if len(odlabels)>0:
            newdata = {label: odlabels}
        else:
            newdata = {label: None}
        self._labels.update(newdata)

    
    def split_data_into_tiles(self, data_type=None, **kwargs):
        labeld = 'tiles'
        data_type = self._check_applyaug2specifictype(data_type, labeld)
        fun = "split_image"
        self._augmentation(fun, data_type, label = labeld, **kwargs)


    def aug_constrast_image(self, data_type=None, samplesize=None,seed = 123,**kwargs):
        labeld = 'contrast'
        data_type = self._check_applyaug2specifictype(data_type, labeld)
        fun = "change_images_contrast"
        self._augmentation(fun, data_type, label = labeld, samplesize=samplesize,seed = seed,**kwargs)

    def aug_clahe_image(self, data_type=None, samplesize=None,seed = 123,**kwargs):
        labeld = 'clahe'
        data_type = self._check_applyaug2specifictype(data_type, labeld)
        fun = "clahe_img"
        self._augmentation(fun, data_type, label = labeld, samplesize=samplesize,seed = seed,**kwargs)


    def aug_rotate_image(self, data_type=None,samplesize=None,seed = 123, **kwargs):
        labeld = 'rotate'
        data_type = self._check_applyaug2specifictype(data_type, labeld)
        fun = "rotate_npimage"
        self._augmentation(fun, data_type, label = labeld, samplesize=samplesize,seed = seed, **kwargs)
    

    def aug_expand_image(self, data_type=None, samplesize=None,seed = 123,**kwargs):
        labeld = 'expand'
        data_type = self._check_applyaug2specifictype(data_type, labeld)
        fun = "expand_npimage"
        
        self._augmentation(fun, data_type, label = labeld, samplesize=samplesize,seed = seed, **kwargs)

    def aug_change_hsv(self, data_type=None,samplesize=None,seed = 123, **kwargs):
        labeld = 'hsv'
        data_type = self._check_applyaug2specifictype(data_type, labeld)
        fun = "shift_hsv"
        
        self._augmentation(fun, data_type, label = labeld, samplesize=samplesize,seed = seed, **kwargs)
        

    def aug_blur_image(self, data_type=None,samplesize=None,seed = 123, **kwargs):
        labeld = 'blur'
        data_type = self._check_applyaug2specifictype(data_type, labeld)
        fun = "blur_image"
        
        self._augmentation(fun, data_type, label = labeld, samplesize=samplesize,seed = seed, **kwargs)


    def plot_image(self, id_image=0, figsize=(12, 10), add_label=False, sourcetype = 'raw'):
        if add_label and self.od_labels[sourcetype] is not None:
            vector = []
            imgsize = (self.images_data[sourcetype][id_image].shape[0], self.images_data[sourcetype][id_image].shape[1])
            for j in range(len(self.od_labels[sourcetype][id_image])):
                vector.append(from_yolo_toxy(
                    [float(i) for i in self.od_labels[sourcetype][id_image][j]],
                    imgsize))

            plot_single_image_odlabel(self.images_data[sourcetype][id_image], vector, figsize=figsize)

        else:

            plot_single_image(self.images_data[sourcetype], id_image, figsize)

    def _read_single_image(self, id_image=0, scale_factor=None, size=None, pattern="\\", pos_id=-2):

        kwargs = {'scale_factor': scale_factor,
                  'size': size}

        if type(id_image) == str:
            id_image = [i for i in range(len(self.jpg_path_files))
                        if id_image in self.jpg_path_files[i]]

        id_image = self.jpg_path_files[id_image].split(pattern)[pos_id:]
        id_image_folder = '_'.join(id_image)
        selectid = 0
        if len(id_image) > 0:
            selectid = -1

        path_img_id = [i for i in self.jpg_path_files if id_image[selectid] in i]

        single_image = load_image_into_numpy_array(path_img_id[0], **kwargs)

        return single_image, id_image_folder

    def _organice_data(self, idx, kwargs):
        img_list = []
        bb_list = []

        img_info = self.single_image(idx, **kwargs)
        for m in range(len(img_info[1])):
            img_list.append(img_info[0][0])
            bb_list.append(img_info[1][m])
        return [img_list, bb_list]

    def multiple_images(self,
                        n_samples=None,
                        scale_factor=None,
                        size=None,
                        shuffle=True,
                        seed=123,
                        start=0,
                        read_images=True):

        kwargs = {'scale_factor': scale_factor,
                  'size': size}
        if n_samples is None:
            n_samples = len(self._path_files)

        list_idx = list(range(len(self._path_files)))
        if shuffle:
            random.seed(seed)
            random.shuffle(list_idx)

        img_list = []
        bb_list = []
        idx_list = []

        for i in tqdm(range(start, n_samples)):

            img_info = self.single_image(list_idx[i], read_images=read_images, **kwargs)
            for m in range(len(img_info[1])):
                img_list.append(img_info[0][0])
                bb_list.append(img_info[1][m])
            idx_list.append(list_idx[i])

        return img_list, bb_list, idx_list

    #def single_image(self, idx, read_images=True, **kwargs):
    #
    #    with open(self._path_files[idx], 'r', encoding="utf8") as f:
    #        data = f.read()

    #   # get image name
    #    bsdata = get_image_path_from_xml(data)
    #    # get boundary box name
    #    bbdata = get_bounding_box_from_xml(data)
    #    # get original_size
    #    sizedata = get_image_orgsize_from_xml(data)
    #    gt_boxes = []
    #    train_images_np = []

    #    for m in range(len(bbdata)):
    #        gt_boxes.append(bb_topercentage(bbdata[m], sizedata))
    #        if read_images:
    #            train_images_np.append(load_image_into_numpy_array(bsdata[0], **kwargs))
    #        else:
    #            train_images_np.append(None)

    #    return [train_images_np, gt_boxes]

    def read_image_data(self, id_images=None, scale_factor=None, size=None, pattern="\\", pos_id=-2):

        kwargs = {'scale_factor': scale_factor, 'size': size, 'pattern': pattern,
                  'pos_id': pos_id}

        images_list = []
        file_names = []
        for i in tqdm(range(len(id_images))):
            imgdata = self._read_single_image(i, **kwargs)
            images_list.append(imgdata[0])
            file_names.append(imgdata[1])

        return images_list, file_names

    def to_jpg(self, output_path, data_type=None, size=None, verbose=False) -> None:

        if data_type is not None:
            data_type = [data_type] if type(data_type) != list else data_type
        else:
            data_type = self._augmented_data.keys()

        ## export labels if there are any
        if self._labels['raw'] is not None:
            if not os.path.isdir(os.path.join(output_path,'images')):
                os.mkdir(os.path.join(output_path,'images'))

            if not os.path.isdir(os.path.join(output_path,'labels')):
                os.mkdir(os.path.join(output_path,'labels'))

        for datatype in data_type:
            for idimage in range(len(self._augmented_data[datatype]['imgs'])):
                fn = self._augmented_data[datatype]['names'][idimage] + '.jpg'
                fn_path = os.path.join(output_path, fn)

                if self._labels['raw'] is not None:
                    if self._labels[datatype] is not None:

                        save_yololabels(self._labels[datatype][idimage],
                                    self._augmented_data[datatype]['names'][idimage] + '.txt' ,
                                    outputdir= os.path.join(output_path,'labels'))
                        if self._labels[datatype][idimage] is not None:
                            from_array_2_jpg(self._augmented_data[datatype]['imgs'][idimage],
                                    os.path.join(output_path,'images', fn),
                                    size=size,
                                    verbose=verbose)
                else:
                    from_array_2_jpg(self._augmented_data[datatype]['imgs'][idimage],
                                 fn_path,
                                 size=size,
                                 verbose=verbose)
                
    def get_image(self, id):
        pass
    
    def __init__(self,
                 source,
                 id_image=None,
                 image_size=None,
                 scale_percentage=None,
                 pattern='jpg',
                 label_type=None,
                 sep_pattern="\\",
                 path_to_images=False):
        """

        :param source:
        :param id_image:
        :param image_size:
        :param scale_percentage:
        :param pattern:
        :param sep_pattern:
        :param path_to_images
        """

        self._path_files = None
        self._input_path = source
        if path_to_images:
            self._path_files = source
            self.jpg_path_files = self._path_files
            self._input_path = "/".join(list(np.array(source[0].split('[/\\]'))[:-1]))

        else:
            if pattern is not None:

                self._path_files = list_files(source, pattern=pattern)
                if label_type == "xml":
                    self.jpg_path_files = [i[:-4] + ".jpg" for i in self._path_files]

                if pattern == 'jpg':
                    self.jpg_path_files = self._path_files

        # TODO: separate images_data and id_image
        if id_image is not None:
            if id_image == "all":
                id_image = list(range(len(self.jpg_path_files)))
            if type(id_image) != list:
                id_image = [id_image]
            self.jpg_path_files = [self.jpg_path_files[i] for i in id_image]
            print(self.jpg_path_files)
        else:
            id_image = list(range(len(self.jpg_path_files)))

        images_data, self.id_image = self.read_image_data(id_images=id_image,
                                                               scale_factor=scale_percentage,
                                                               size=image_size,
                                                               pattern=sep_pattern, pos_id=-2)

        fn_originals = [get_filename_frompath(self.jpg_path_files[i])
                        for i in range(len(self.jpg_path_files))]

        self._augmented_data = {'raw': {'imgs': images_data,
                                        'names': fn_originals}}

        #try:
        alllabels = LabelData(self).labels
        nnone = len([lab for lab in alllabels if lab is None])

        if nnone ==  len(images_data):
            alllabels = None

        self._labels = {'raw': alllabels}
        #except:
        #    self._labels = None


"""
    #def object_detection(self, model,
    #                     cat_index,

    #                     min_score=0.55,
                         fig_size=(15, 20),
                         output_path=None,
                         plot=False
                         ):

        detections = []

        for i in tqdm(range(len(self.images_data))):
            if output_path is not None:
                fn = os.path.join(output_path, "n_{}.jpg".format(self.id_image[i]))

            else:
                fn = None

            detections.append(
                single_image_detection(self.images_data[i],
                                       model,
                                       cat_index,
                                       filename=fn,
                                       min_score=min_score,
                                       fig_size=fig_size,
                                       plot=plot))
        return detections
"""
