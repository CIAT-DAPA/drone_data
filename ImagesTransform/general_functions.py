import random
import numpy as np
import cv2
import os
import glob

def get_ids_split_datasets(len_data, val_perc=20, shuffle=True, seed=123, test_perc=None):

    list_idx = list(range(len_data))
    if shuffle:
        random.seed(seed)
        random.shuffle(list_idx)

    val_len = int(len_data * (val_perc / 100))
    n_training = len_data - val_len
    if test_perc is None:
        val_ids = np.array(list_idx)[n_training:]
        train_ids = np.array(list_idx)[:n_training]
        output = [train_ids, val_ids]
    else:
        test_len = int(len_data * (test_perc / 100))
        n_training = n_training - test_len
        val_ids = np.array(list_idx)[n_training:(n_training + val_len)]
        test_ids = np.array(list_idx)[(n_training + val_len):]
        train_ids = np.array(list_idx)[:n_training]
        output = [train_ids, val_ids, test_ids]

    return output

def check_folder(folderpath, verbose = False):

    if folderpath is not None:
        if not os.path.exists(folderpath):
            os.mkdir(folderpath)
            if verbose: 
                print("following path was created {}".format(folderpath))
    else:
        folderpath = ""
    return folderpath


## taken from https://medium.com/@iKhushPatel/convert-video-to-images-images-to-video-using-opencv-python-db27a128a481


def from_video_toimages(video_path, outputpath = None, frame_rate = 5, preffix = "image"):

    outputpath = check_folder(outputpath)

    vidcap = cv2.VideoCapture(video_path)
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            cv2.imwrite(os.path.join(outputpath, preffix+str(count)+".jpg"), image)     # save frame as JPG file
        return hasFrames

    sec = 0

    count=1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frame_rate
        sec = round(sec, 2)
        success = getFrame(sec)


def get_filename_frompath(imgpath, pattern = 'jpg'):
    imgfilename = imgpath.split('\\')[-1]

    if imgfilename.endswith(pattern):
        imgfilename = imgfilename[:-4]

    return imgfilename


def list_files(input_path, pattern="xml"):
    """

    :param input_path:
    :param pattern:
    :return:
    """
    # taken from https://www.askpython.com/python/examples/list-files-in-a-directory-using-python

    files = glob.glob(input_path + '**/*' + pattern, recursive=True)
    return [f for f in files if os.path.isfile(f)]
