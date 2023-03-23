

from pathlib import Path
import numpy as np
import geopandas as gpd
import pandas as pd
import torch
import time
import torchvision
import cv2
import copy

from . import gis_functions as gf
from .data_processing import from_xarray_2array
from .data_processing import find_date_instring
from .drone_data import DroneData
from .gis_functions import merging_overlaped_polygons, from_bbxarray_2polygon
from .data_processing import resize_3dnparray

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
    

def draw_frame(img, bbbox, dictlabels = None, default_color = (255,255,255)):
    imgc = img.copy()
    for i in range(len(bbbox)):
        x1,x2,y1,y2 = bbbox[i]

        widhtx = abs(x1 - x2)
        heighty = abs(y1 - y2)

        start_point = (x1, y1)
        end_point = (x2,y2)
        if dictlabels is not None:
            color = dictlabels[i]['color']
            label = dictlabels[i]['label']
        else:
            label = ''
            color = default_color

        thickness = 4
        xtxt = x1 if x1 < x2 else x2
        ytxt = y1 if y1 < y2 else y2
        imgc = cv2.rectangle(imgc, start_point, end_point, color, thickness)
        if label != '':

            imgc = cv2.rectangle(imgc, (xtxt,ytxt), (xtxt + int(widhtx*0.8), ytxt - int(heighty*.2)), color, -1)
            
            imgc = cv2.putText(img=imgc, text=label,org=(xtxt + int(abs(x1-x2)/15),
                                                            ytxt - int(abs(y1-y2)/20)), 
                                                            fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1*((heighty)/200), color=(255,255,255), thickness=2)
            
    return imgc   

def check_image(img, inputshape = (512,512)):

    imgc = img.copy()

    if len(imgc.shape) == 3:
        imgc = np.expand_dims(imgc, axis=0)

    if imgc.shape[3] == 3:
        if (not imgc.shape[2] == inputshape[0]) and not (imgc.shape[3] == inputshape[1]):
            imgc = cv2.resize(imgc[0], inputshape, interpolation=cv2.INTER_AREA)
            imgc = np.expand_dims(imgc, axis=0)
        imgc = imgc.swapaxes(3, 2).swapaxes(2, 1)
    else:
        imgc = imgc.swapaxes(1, 2).swapaxes(2, 3)
        if (not imgc.shape[2] == inputshape[0]) and not (imgc.shape[3] == inputshape[1]):
            imgc = cv2.resize(imgc[0], inputshape, interpolation=cv2.INTER_AREA)
        imgc = imgc.swapaxes(3, 2).swapaxes(2, 1)

    return imgc

class DroneObjectDetection(DroneData):
    
    def draw_bb_in_tile(self,imgtile):
        xyposhw,yoloimgcoords = self.predict_tile_coords(imgtile, conf_thres=0.50)
        m= []
        for l, r, t, b in yoloimgcoords:
            m.append([l, r, t, b])

        imgdraw = draw_frame(imgtile.copy().to_array().values.swapaxes(0,1).swapaxes(1,2), m)
        return imgdraw


    def predict_tile_coords(self, imgtile, **kwargs):

        img0 = imgtile.copy().to_array().values
        output = None
        yolocoords = []
        if not np.isnan(img0.sum()) and img0.shape[1] == img0.shape[2]:
        
            bbpredictions, img1 = self.predict(img0, **kwargs)
            if img0.shape[0] == 3:
                img0 = img0.swapaxes(0, 1).swapaxes(1, 2)
            xyxylist,yolocoords = xyxy_predicted_box(bbpredictions, img0.shape, img1.shape)

            ### save as shapefiles
            crs_system = imgtile.attrs['crs']
            polsshp_list = []
            
            if len(xyxylist):
                for i in range(len(xyxylist)):
                    bb_polygon = from_bbxarray_2polygon(xyxylist[i][0], imgtile)

                    pred_score = np.round(xyxylist[i][2] * 100, 3)

                    gdr = gpd.GeoDataFrame({'pred': [i],
                                            'score': [pred_score],
                                            'geometry': bb_polygon},
                                        crs=crs_system)

                    polsshp_list.append(gdr)
                output = pd.concat(polsshp_list, ignore_index=True)

        return output, yolocoords

    def detect_oi_in_uavimage(self, imgsize = 512, overlap = None, aoi_limit = 0.5, onlythesetiles = None, **kwargs):
        """
        a function to detect opbect of interest in a RGB UAV image

        parameters:
        ------
        imgpath: str:
        """
        overlap = [0] if overlap is None else overlap
        allpols_pred = []
        for spl in overlap:
            self.split_into_tiles(width = imgsize, height = imgsize, overlap = spl) 
            if onlythesetiles is not None:
                tileslist =  onlythesetiles
            else:
                tileslist =  list(range(len(self._tiles_pols)))

            for i in tileslist:
                print(i)

                
                bbasgeodata, _ = self.predict_tile_coords(self.tiles_data(i), **kwargs)
                
                if bbasgeodata is not None:
                    bbasgeodata['tile']= [i for j in range(bbasgeodata.shape[0])]
                    allpols_pred.append(bbasgeodata)

        allpols_pred_gpd = pd.concat(allpols_pred)
        allpols_pred_gpd['id'] = [i for i in range(allpols_pred_gpd.shape[0])]

        #allpols_pred_gpd.to_file("results/alltest_id.shp")
        print("{} polygons were detected".format(allpols_pred_gpd.shape[0]))

        total_objects = merging_overlaped_polygons(allpols_pred_gpd, aoi_limit = aoi_limit)
        total_objects = merging_overlaped_polygons(pd.concat(total_objects), aoi_limit = aoi_limit)
        total_objects = pd.concat(total_objects) 
        print("{} boundary boxes were detected".format(total_objects.shape[0]))
        
        return total_objects

    
    def predict(self, image, conf_thres=0.5,
                       iou_thres=0.45,
                       classes=None,
                       agnostic_nms=False,
                       half = False,
                       max_det=1000):

        
        imgc = check_image(image)
        img = torch.from_numpy(imgc).to(self.device)
        img = img.half() if half else img.float()
        
        img = img / 255.
        print(img.shape)
        bounding_box = self.model(img, augment=False)
        pred = non_max_suppression(bounding_box, conf_thres, iou_thres, classes,
                               agnostic_nms, max_det=max_det)
        return pred, img

    def __init__(self, inputpath, yolo_model = None, device = None, **kwargs) -> None:
        
        super().__init__(
                 inputpath,
                 **kwargs)

        self.device = device
        self.model = yolo_model


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



def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)



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


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    if isinstance(prediction, (list, tuple)):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = 'mps' in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(x[:, :4])  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        
        #if (time.time() - t) > time_limit:
        #    LOGGER.warning(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
        #    break  # time limit exceeded

    return output

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

"""

DEPRECATED
def xyxy_predicted_box(img, yolo_model, device, half = False,
                       conf_thres=0.5,
                       iou_thres=0.45,
                       classes=None,
                       agnostic_nms=False,
                       max_det=1000):

    imgc = img.copy()
    if img.shape[0] != 3:
        img = img.swapaxes(2,1).swapaxes(1,0)
    
    img = torch.from_numpy(img).to(device)

    img = img.half() if half else img.float()
    img /= 255.0
    if len(img.shape) == 3:
        img = img[None]
    
    pred = yolo_model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes,
                               agnostic_nms, max_det=max_det)

    
    xyxylist = []
    yolocoords = []
    if imgc.shape[0] == 3:
        img0 = imgc.swapaxes(0,1).swapaxes(1,2)
    else:
        img0 = imgc
    
    for i, det in enumerate(pred):
        s, im0 = '', img0
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        print(img.shape[2:])
        if len(det):
            print(det[:, :4])
            #det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()
            print(det)
            for *xyxy, conf, cls in det:
                # Rescale boxes from img_size to im0 size
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                xyxylist.append([torch.tensor(xyxy).tolist(), xywh, conf.tolist()])
                m = [0]
                for i in range(len(xywh)):
                    m.append(xywh[i])
                    
                l, r, t, b = from_yolo_toxy(m, (im0.shape[0],
                                            im0.shape[1]))

                yolocoords.append([l, r, t, b])

    return xyxylist,yolocoords


"""

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