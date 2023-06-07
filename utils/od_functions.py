

from pathlib import Path
import numpy as np
import geopandas as gpd
import pandas as pd
import torch
import time
import torchvision
import cv2
import copy
import tqdm

from . import gis_functions as gf
from .data_processing import from_xarray_2array
from .data_processing import find_date_instring
from .drone_data import DroneData
from .gis_functions import merging_overlaped_polygons, from_bbxarray_2polygon
from .data_processing import resize_3dnparray
import os

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


def bb_as_dataframe(xarraydata, yolo_model, device, half =False,
                       conf_thres=0.70, img_size=512, min_size=128,
                       bands=['red', 'green', 'blue']):
                       
    ind_data = xarraydata[bands].copy().to_array().values
    imgsz = ind_data.shape[1] if ind_data.shape[1] < ind_data.shape[2] else ind_data.shape[2]

    output = None

    if imgsz >= min_size:

        if (img_size - imgsz) > 0:
            ind_data = resize_3dnparray(ind_data, img_size)

        bb_predictions = xyxy_predicted_box(ind_data, yolo_model, device, half, conf_thres)

        ### save as shapefiles
        crs_system = xarraydata.attrs['crs']
        polsshp_list = []
        if len(bb_predictions):
            for i in range(len(bb_predictions)):
                bb_polygon = from_bbxarray_2polygon(bb_predictions[i][0], xarraydata)

                pred_score = np.round(bb_predictions[i][2] * 100, 3)

                gdr = gpd.GeoDataFrame({'pred': [i],
                                        'score': [pred_score],
                                        'geometry': bb_polygon},
                                       crs=crs_system)

                polsshp_list.append(gdr)
            output = pd.concat(polsshp_list, ignore_index=True)

    return output,bb_predictions


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes
def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def xyxy_predicted_box(bbpredicted, im0shape, img1shape):

    pred = bbpredicted

    #print(pred)
    xyxylist = []
    yolocoords = []
    
    for i, det in enumerate(pred):
        #s, im0 = '', img0
        gn = torch.tensor(im0shape)[[1, 0, 1, 0]]
        if len(det):
            
            #det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            det[:, :4] = scale_boxes(img1shape[2:], det[:, :4], im0shape).round()
            
            for *xyxy, conf, cls in det:
                # Rescale boxes from img_size to im0 size
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                xyxylist.append([torch.tensor(xyxy).tolist(), xywh, conf.tolist()])
                m = [0]
                for i in range(len(xywh)):
                    m.append(xywh[i])
                    
                l, r, t, b = from_yolo_toxy(m, (im0shape[0],
                                            im0shape[1]))

                yolocoords.append([l, r, t, b])

    return xyxylist,yolocoords
    

#from od_data_awaji_2022.yolov5_master.models.experimental import attempt_load
#from od_data_awaji_2022.yolov5_master.utils.torch_utils import select_device
#from yolo_utils.general import non_max_suppression, scale_coords, set_logging, xyxy2xywh

"""
@torch.no_grad()
def load_weights_model(wpath, device='', half=False):
    set_logging()
    device = select_device(device)

    half &= device.type != 'cpu'
    w = str(wpath[0] if isinstance(wpath, list) else wpath)

    model = torch.jit.load(w) if 'torchscript' in w else attempt_load(wpath, device=device)

    if half:
        model.half()  # to FP16

    return [model, device, half]
"""

### the following functions were taken from yolov5 




def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[:, [0, 2]] -= pad[0]  # x padding
    boxes[:, [1, 3]] -= pad[1]  # y padding
    boxes[:, :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes
    

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2






def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2



<<<<<<< HEAD
=======
def xyxy2xywhn(x, w=640, h=640, clip=False, eps=0.0):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] normalized where xy1=top-left, xy2=bottom-right
    if clip:
        clip_boxes(x, (h - eps, w - eps))  # warning: inplace clip
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = ((x[:, 0] + x[:, 2]) / 2) / w  # x center
    y[:, 1] = ((x[:, 1] + x[:, 3]) / 2) / h  # y center
    y[:, 2] = (x[:, 2] - x[:, 0]) / w  # width
    y[:, 3] = (x[:, 3] - x[:, 1]) / h  # height
    return y

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


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

>>>>>>> 0ad447aecb5d1a494f0ebd486ff9607baf39e781
def odboxes_per_xarray(xarraydata, yolo_model, device, half,
                       conf_thres=0.70, img_size=512, min_size=128,
                       bands=['red', 'green', 'blue']):
                       
    ind_data = xarraydata[bands].copy().to_array().values
    imgsz = ind_data.shape[1] if ind_data.shape[1] < ind_data.shape[2] else ind_data.shape[2]

    output = None

    if imgsz >= min_size:

        if (img_size - imgsz) > 0:
            ind_data = resize_3dnparray(ind_data, img_size)

        bb_predictions = xyxy_predicted_box(ind_data, yolo_model, device, half, conf_thres)

        ### save as shapefiles
        crs_system = xarraydata.attrs['crs']
        polsshp_list = []
        if len(bb_predictions):
            for i in range(len(bb_predictions)):
                bb_polygon = gf.from_bbxarray_2polygon(bb_predictions[i][0], xarraydata)

                pred_score = np.round(bb_predictions[i][2] * 100, 3)

                gdr = gpd.GeoDataFrame({'pred': [i],
                                        'score': [pred_score],
                                        'geometry': bb_polygon},
                                       crs=crs_system)

                polsshp_list.append(gdr)
            output = pd.concat(polsshp_list, ignore_index=True)

    return output,bb_predictions


def detect_oi_in_uavimage(drone_data, model, device, imgsize = 512, conf_thres = 0.65, aoi_limit = 0.5, roi = None, overlap = [0.25, 0.40]):
    
    if type(drone_data) is str:
        fielddata  = drone_data.DroneData(drone_data, multiband_image=True,roi=roi)
        date = find_date_instring(drone_data)
    else:
        fielddata = copy.deepcopy(drone_data)

    
    allpols_pred= []

    for spl in overlap:
        
        fielddata.split_into_tiles(width = imgsize, height = imgsize, overlap = spl) 

        for i in range(len(fielddata._tiles_pols)):

            poltile_predictions, pred = odboxes_per_xarray(fielddata.tiles_data(i), 
                                                    model, device, 
                                                    half = False, 
                                                    img_size = imgsize,
                                                    conf_thres= conf_thres)  
            
            if poltile_predictions is not None:
                poltile_predictions['tile']= [i for j in range(poltile_predictions.shape[0])]
                poltile_predictions['date']= date
                allpols_pred.append(poltile_predictions)

    allpols_pred_gpd = pd.concat(allpols_pred)
    allpols_pred_gpd['id'] = [i for i in range(allpols_pred_gpd.shape[0])]

    #allpols_pred_gpd.to_file("results/alltest_id.shp")
    print("{} polygons were detected".format(allpols_pred_gpd.shape[0]))

    total_objects = merging_overlaped_polygons(allpols_pred_gpd, aoi_limit = aoi_limit)
    total_objects = merging_overlaped_polygons(pd.concat(total_objects), aoi_limit = aoi_limit)
    total_objects = pd.concat(total_objects) 
    print("{} boundary boxes were detected".format(total_objects.shape[0]))
    return total_objects