from .utils import *
import torch

from ..utils.drone_data import DroneData
from ..utils.gis_functions import from_bbxarray_2polygon, merging_overlaped_polygons

from .plt_utils import draw_frame
import geopandas as gpd
import pandas as pd
import numpy as np
import tqdm

class DroneObjectDetection(DroneData):
    
    """class to detect objects using a YOLOV5 model
    """
    
    def draw_bb_in_tile(self,imgtile, conf_thres=0.50):
        xyposhw,yoloimgcoords = self.predict_tile_coords(imgtile, conf_thres=conf_thres)
        m= []
        for l, r, t, b in yoloimgcoords:
            m.append([l, r, t, b])

        imgdraw = draw_frame(imgtile.copy().to_array().values.swapaxes(0,1).swapaxes(1,2), m)
        return imgdraw

    def predict_image(self, bands = ['red','green','blue'], **kwargs):
        
        img0 = self.drone_data[bands].copy().to_array().values
        
        output = None
        yolocoords = []
        if not np.isnan(img0.sum()):
            bbpredictions, img1 = self.predict(img0, **kwargs)
            
            if img0.shape[0] == 3:
                img0 = img0.swapaxes(0, 1).swapaxes(1, 2)
            xyxylist,yolocoords = xyxy_predicted_box(bbpredictions, img0.shape, img1.shape)
            
            ### save as shapefiles
            crs_system = self.drone_data.attrs['crs']
            polsshp_list = []
            
            if len(xyxylist):
                for i in range(len(xyxylist)):
                    bb_polygon = from_bbxarray_2polygon(xyxylist[i][0], self.drone_data)

                    pred_score = np.round(xyxylist[i][2] * 100, 3)

                    gdr = gpd.GeoDataFrame({'pred': [i],
                                            'score': [pred_score],
                                            'geometry': bb_polygon},
                                        crs=crs_system)

                    polsshp_list.append(gdr)
                output = pd.concat(polsshp_list, ignore_index=True)

        return output, yolocoords
        
    
    def predict_tile_coords(self, imgtile, bands = ['blue', 'green','red'], **kwargs):

        img0 = imgtile[bands].copy().to_array().values
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

            for i in tqdm.tqdm(tileslist):
               
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
        
        bounding_box = self.model(img, augment=False)
        pred = non_max_suppression(bounding_box, conf_thres, iou_thres, classes,
                               agnostic_nms, max_det=max_det)
        return pred, img
    
    def export_as_yolo_training(self):
        
        pass
    def __init__(self, inputpath, yolo_model = None, device = None, **kwargs) -> None:
        
        super().__init__(
                 inputpath,
                 **kwargs)

        self.device = device
        self.model = yolo_model
        