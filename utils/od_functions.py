from utils import gis_functions as gf
from utils.data_processing import from_xarray_2array
from utils.data_processing import resize_3dnparray
from pathlib import Path
import numpy as np
import geopandas as gpd
import pandas as pd
import torch

from models.experimental import attempt_load

from yolo_utils.torch_utils import select_device
from yolo_utils.general import non_max_suppression, scale_coords, set_logging, xyxy2xywh


@torch.no_grad()
def load_weights_model(wpath, device='', half=False):
    set_logging()
    device = select_device(device)

    half &= device.type != 'cpu'
    w = str(wpath[0] if isinstance(wpath, list) else wpath)

    model = torch.jit.load(w) if 'torchscript' in w else attempt_load(wpath, map_location=device)

    if half:
        model.half()  # to FP16

    return [model, device, half]


def xyxy_predicted_box(img, yolo_model, device, half,
                       conf_thres=0.5,
                       iou_thres=0.45,
                       classes=None,
                       agnostic_nms=False,
                       max_det=1000):
    imgc = img.swapaxes(1, 0).swapaxes(2, 1)[:, :, [2, 1, 0]]
    img = torch.from_numpy(img).to(device)

    img = img.half() if half else img.float()
    img /= 255.0
    if len(img.shape) == 3:
        img = img[None]

    pred = yolo_model(img, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes,
                               agnostic_nms, max_det=max_det)
    xyxylist = []
    for i, det in enumerate(pred):
        s, im0 = '', imgc.copy()
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in det:
                # Rescale boxes from img_size to im0 size
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                xyxylist.append([torch.tensor(xyxy).tolist(), xywh, conf.tolist()])

    return xyxylist


def odboxes_per_xarray(xarraydata, yolo_model, device, half,
                       conf_thres=0.70, img_size=512, min_size=128,
                       bands=['red', 'green', 'blue']):
    ind_data = from_xarray_2array(xarraydata, bands)

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

    return output
