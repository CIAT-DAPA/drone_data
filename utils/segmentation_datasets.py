from Crop_DL.crop_dl.seeds.utils import getmidleheightcoordinates, euclidean_distance, getmidlewidthcoordinates
from Crop_DL.crop_dl.image_functions import contours_from_image, pad_images
from Crop_DL.crop_dl.plt_utils import random_colors, add_frame_label
from Crop_DL.crop_dl.dataset_utils import get_boundingboxfromseg

from Crop_DL.crop_dl.models.utils import image_to_tensor
from Crop_DL.crop_dl.models.dl_architectures import Unet256

import copy
import geopandas as gpd
import pandas as pd
import math
import xarray
import torch
import torch.optim as optim
import cv2
import numpy as np
import tqdm

from .drone_data import DroneData
from .data_processing import from_xarray_2array
from .gis_functions import merging_overlaped_polygons
from .decorators import check_output_fn

from ..uavdl.plt_utils import plot_segmenimages

def get_clossest_prediction(image_center, bb_predictions, distance_limit = 30):
    distpos = None
    dist = []
    if bb_predictions is not None:
        if len(bb_predictions)>0:
            for i in range(len(bb_predictions)):
                x1,y1,x2,y2 = bb_predictions[i]
                widthcenter = (x1+x2)//2
                heightcenter = (x1+x2)//2
                dist.append(euclidean_distance([widthcenter,heightcenter],image_center))
            
            if np.min(dist)<distance_limit:
                distpos = np.where(np.array(dist) == np.min(dist))[0][0]

    return distpos, dist

def _apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

def find_contours(image, hull = False):
    
    maskimage = image.copy()
    #imgmas = (maskimage*255).astype(np.uint8)
    ## mask must has alues btwn 0  and 255
    if np.max(maskimage)==1.:
        maskimage = maskimage * 255.
    contours = contours_from_image(maskimage)
    if hull:
        firstcontour = cv2.convexHull(contours[0])
    else:
        firstcontour = contours[0]
        
    rect = cv2.minAreaRect(firstcontour)
    
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box
    

def clip_image(image, bounding_box, 
               bbtype = 'xminyminxmaxymax', 
               padding = None, 
                    paddingwithzeros =True):
        
    if bbtype == 'xminyminxmaxymax':
        x1,y1,x2,y2 = bounding_box
        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
        
    if padding:
        if paddingwithzeros:
            imgclipped = image[
            y1:y2,x1:x2] 
            
            imgclipped = pad_images(imgclipped, padding_factor = padding)
        else:
            height = abs(y1-y2)
            width = abs(x1-x2)
            zoom_factor = padding / 100 if padding > 1 else padding
            new_height, new_width = height + int(height * zoom_factor), width + int(width * zoom_factor)  
            pad_height1, pad_width1 = abs(new_height - height) // 2, abs(new_width - width) //2
            newy1 = 0 if (y1 - pad_height1)<0 else (y1 - pad_height1)
            newx1 = 0 if (x1 - pad_width1)<0 else (x1 - pad_width1)
            imgclipped = image[newy1:newy1+(height+pad_height1*2), 
                                newx1:newx1+(width+pad_width1*2)] 
    
    else:
        imgclipped = image[
            y1:y2,x1:x2] 
    
    return imgclipped

#
def get_heights_and_widths(maskcontours):

    p1,p2,p3,p4=maskcontours
    alpharad=math.acos((p2[0] - p1[0])/euclidean_distance(p1,p2))

    pheightu=getmidleheightcoordinates(p2,p3,alpharad)
    pheigthb=getmidleheightcoordinates(p1,p4,alpharad)
    pwidthu=getmidlewidthcoordinates(p4,p3,alpharad)
    pwidthb=getmidlewidthcoordinates(p1,p2,alpharad)

    return pheightu, pheigthb, pwidthu, pwidthb


def plot_individual_mask(rgbimg, maskimg, textlabel = None,
                         col = [0,255,255], mask_image = True,
                         addlines = True, addlabel = True,
                         addmask = True,
                         addsquarecontour = True,
                         alpha = 0.2,
                         sizefactorred = 250,
                         
                    heightframefactor = .15,
                    widthframefactor = .3,
                    textthickness = 1):
    

    msksones = maskimg.copy()

    if np.max(msksones != 1):
        msksones[msksones<150] = 0
        msksones[msksones>=150] = 1
    
    if mask_image:
        newimg = cv2.bitwise_and(rgbimg.astype(np.uint8),rgbimg.astype(np.uint8),
                                mask = msksones)
    else:
        newimg = np.array(rgbimg).astype(np.uint8)

    if addmask:
        img = _apply_mask(newimg, (msksones).astype(np.uint8), col, 
                      alpha=alpha)
    else:
        img = newimg
    #img = newimg
    
    linecolor = list((np.array(col)*255).astype(np.uint8))
    m = np.ascontiguousarray(img, dtype=np.uint8)
    
    if addsquarecontour:
        m = cv2.drawContours(m,[find_contours(maskimg, hull = True)],0,[int(i) for i in linecolor],1)
    
    if addlines:
        pheightu, pheigthb, pwidthu, pwidthb = get_heights_and_widths(
            find_contours(maskimg, hull = True))
        m = cv2.line(m, pheightu, pheigthb, (0,0,0), 1)
        m = cv2.line(m, pwidthu, pwidthb, (0,0,0), 1)

    if addlabel:
        str_id = textlabel if textlabel is not None else '0'

        x1,y1,x2,y2 = get_boundingboxfromseg(maskimg)

        m = add_frame_label(m,
                str(str_id),
                [int(x1),int(y1),int(x2),int(y2)],[
            int(i*255) for i in col],
                sizefactorred = sizefactorred,
                heightframefactor = heightframefactor,
                widthframefactor = widthframefactor,
                textthickness = textthickness)
    return m


def get_boundingboxfromseg(mask):
    
    pos = np.where(mask)
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    
    return([xmin, ymin, xmax, ymax])

def get_height_width(grayimg, pixelsize = 1):
    
    ## find contours
    wrapped_box = find_contours(grayimg)

    ## get distances
    pheightu, pheigthb, pwidthu, pwidthb = get_heights_and_widths(wrapped_box)
    d1 = euclidean_distance(pheightu, pheigthb)
    d2 = euclidean_distance(pwidthu, pwidthb)
    
    pheightu, pheigthb, pwidthu, pwidthb = get_heights_and_widths(wrapped_box)
    d1 = euclidean_distance(pheightu, pheigthb)
    d2 = euclidean_distance(pwidthu, pwidthb)

    ## with this statement there is an assumption that the rice width is always lower than height
    larger = d1 if d1>d2 else d2
    shorter = d1 if d1<d2 else d2
    
    return larger* pixelsize, shorter*pixelsize

### isntance segmentation
class SegmentationUAVData(DroneData):
    """Apply instance segmentation models on UAV data

    Args:
        DroneData (_type_): _description_
    """
    
    
    def __init__(self,
                 model,
                 imagepath = None,
                 uavimagery = None,
                 inputsize = (512, 512),
                 tiles_size = (256, 256),
                 tiles_overlap = 0,
                 device = None,
                 multiband_image=True,
                 rgbbands = ["red","green","blue"],
                 spatial_boundary = None) -> None:
        
        import xarray
        self.layer_predictions = {}
        
        if isinstance(uavimagery,xarray.Dataset):
            self.drone_data = uavimagery
        elif imagepath is not None:
            super().__init__(imagepath,multiband_image=multiband_image, bounds = spatial_boundary)
        
            if tiles_size is not None:
                self.split_into_tiles(width = tiles_size[0], height = tiles_size[1], overlap = tiles_overlap) 
            #self.uav_imagery = .drone_data
            
        self.input_model_size = inputsize
        self.model = model
        self._frames_colors = None
        self.tiles_size = tiles_size
        self.rgbbands = rgbbands
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
    
    def _original_size(self, orig_size = None):
        msksc = [0]* len(list(self.msks))
        
        bbsc = [0]* len(list(self.bbs))
        
        for i in range(len(self.msks)):
            msksc[i] = cv2.resize(self.msks[i], 
                                  [orig_size[1],orig_size[0]], 
                                  interpolation = cv2.INTER_AREA)  
            if len(bbsc)>0:
                bbsc[i] = get_boundingboxfromseg(msksc[i])
            #else:
            #    bbsc[i] = []
        
        self.msks = np.array(msksc)
        self.bbs = np.array(bbsc)
        
             
    def _filter_byscore(self, threshold):
        
        pred = self.predictions[0] 
        onlythesepos = np.where(
            pred['scores'].to('cpu').detach().numpy()>threshold)
        
        msks = pred['masks'].mul(255).byte().cpu().numpy()[onlythesepos, 0].squeeze()
        bbs = pred['boxes'].cpu().detach().numpy()[onlythesepos]
        
        if msks.shape[0] == 0:
            msks = np.zeros(self.input_model_size)
            
        if len(msks.shape)==2:
            msks = np.expand_dims(msks,0)
                
        return msks, bbs
    
    def uav_image_prediction(self,uavimage = None, threshold = 0.75, segment_threshold = 180):
        
        if uavimage is None:
            uavimage = self.drone_data.copy()
        
        self.msks = None
        self.bbs = None
        self.scores = None
        
        if np.nansum(uavimage.to_array().values)>0:
            img = from_xarray_2array(uavimage, self.rgbbands , True)
            imgsize = img.copy().shape[1:]
            
            self._img = img
            
            imgtensor = image_to_tensor(img.copy(), self.input_model_size)
            imgtensor.shape
            self.model.eval()
            with torch.no_grad():
                prediction = self.model([imgtensor.to(self.device)])
                
            self.predictions = prediction
            self.idimg = id
            self._imgastensor = imgtensor
            self.scores  = self.predictions[0]['scores'].to('cpu').detach().numpy()
            
            pred = self._filter_byscore(threshold)
            self.bbs = pred[1]
            self.msks = pred[0]
            
            for i in range(len(self.msks)):
                self.msks[i][self.msks[i]<segment_threshold] = 0
            
            self._original_size(imgsize)
            
           
        return {'masks': self.msks, 
                'bbs':self.bbs, 
                'scores':self.scores}
    
    def predict_single_tile(self, id_tile, threshold = 0.75, segment_threshold = 180):
        
        tiledata = self.tiles_data(id_tile)
        height, width = len(tiledata.y),len(tiledata.x)
        ratio = np.min([height/width,width/height])

        self._currenttile = id_tile

        predictions = {'masks': None, 
                'bbs':None, 
                'scores':None}
        if ratio>0.9:
            predictions = self.uav_image_prediction(tiledata, threshold = threshold, 
                                      segment_threshold = segment_threshold)
            
        self.layer_predictions[str(id_tile)] = {'masks': self.msks,
                                                    'bbs':self.bbs}
        return predictions
        
    def plot_prediction(self, **kwargs):
        
        plotimage = None
        
        
        if self.bbs is not None:
            
            if self.bbs is not None:
                bbs= self.bbs.astype(np.uint16)
                if self._frames_colors is None:
                    self._frames_colors = random_colors(len(self.bbs))
            #if id_tile is None:
            #    id_tile = self._currenttile
            #img = from_xarray_2array(self.tiles_data(self._currenttile), ["red","green","blue"], True)
            img = copy.deepcopy(self._img)
            if self.msks.shape[0] == 0:
                msksone = np.zeros(img.shape[:2])
            else:
                msksone = np.max(self.msks, axis = 0)
                
            plotimage =  plot_segmenimages(img.swapaxes(0,1).swapaxes(
                1,2).astype(np.uint8),
                  np.max(np.stack(
                    self.msks), axis = 0)*255, 
                        boxes=self.bbs, 
                        bbtype = 'xminyminxmaxymax',
                        default_color = self._frames_colors,
                        inverrgbtorder=False)
            
        print(self._frames_colors)
        return plotimage
    
    def tile_predictions_asgpd(self, id_tile, **kwargs):

        tile_predictions = self.predict_single_tile(id_tile, **kwargs)
        output = None
        if tile_predictions['bbs'] is not None:
            tileimg = self.tiles_data(self._currenttile)
            crs_system  = tileimg.attrs['crs']
            polsshp_list= []
            for i in range(len(tile_predictions['bbs'])):
                from drone_data.utils.gis_functions import from_bbxarray_2polygon
                
                bb_polygon = from_bbxarray_2polygon(tile_predictions['bbs'][i], tileimg)

                pred_score = np.round(tile_predictions['scores'][i] * 100, 3)

                gdr = gpd.GeoDataFrame({'pred': [i],
                                        'score': [pred_score],
                                        'geometry': bb_polygon},
                                    crs=crs_system)

                polsshp_list.append(gdr)
            if len(polsshp_list):
                output = pd.concat(polsshp_list, ignore_index=True)

        return output
    
    
    def detect_oi_in_uavimage(self, overlap = None, aoi_limit = 0.5, 
                              onlythesetiles = None,threshold = 0.8, **kwargs):
        """
        a function to detect opbect of interest in a RGB UAV image

        parameters:
        ------
        imgpath: str:
        """
        overlap = [0] if overlap is None else overlap
        peroverlap = []
        for spl in overlap:
            allpols_pred = []
            print(f"split overlap {spl}")
            self.split_into_tiles(width = self.tiles_size[0], height = self.tiles_size[1], overlap = spl) 
            
            if onlythesetiles is not None:
                tileslist =  onlythesetiles
            else:
                tileslist =  list(range(len(self._tiles_pols)))

            for i in tqdm.tqdm(tileslist):
                
                #tile_predictions = self.predict_single_tile(i, threshold=threshold, **kwargs)
                
                bbasgeodata = self.tile_predictions_asgpd(i, threshold=threshold, **kwargs)
                
                if bbasgeodata is not None:
                    bbasgeodata['tile']= [i for j in range(bbasgeodata.shape[0])]
                    allpols_pred.append(bbasgeodata)
            
            allpols_pred_gpd = pd.concat(allpols_pred)
            allpols_pred_gpd['id'] = [i for i in range(allpols_pred_gpd.shape[0])]
            total_objects = merging_overlaped_polygons(allpols_pred_gpd, aoi_limit = aoi_limit)
            peroverlap.append(pd.concat(total_objects))

        allpols_pred_gpd = pd.concat(peroverlap)
        allpols_pred_gpd['id'] = [i for i in range(allpols_pred_gpd.shape[0])]
        #allpols_pred_gpd.to_file("results/alltest_id.shp")
        print("{} polygons were detected".format(allpols_pred_gpd.shape[0]))
        
        total_objects = merging_overlaped_polygons(allpols_pred_gpd, aoi_limit = aoi_limit)
        total_objects = pd.concat(total_objects) 
        print("{} boundary boxes were detected".format(total_objects.shape[0]))
        
        return total_objects
    
    def calculate_onecc_metrics(self, cc_id, padding = 20, hull = True):

        # extract only the area that cover the prediction
        maskimage = self._clip_image(self.msks[cc_id], self.bbs[cc_id], padding = padding)
        
        larger, shorter =get_height_width(maskimage.astype(np.uint8))
        msksones = maskimage.copy()
        msksones[msksones>0] = 1
        
        area = np.sum(msksones*1.)

        return {
            'seed_id':[cc_id],'height': [larger], 
                'width': [shorter], 'area': [area]}
        
        
    def plot_individual_cc(self, cc_id,**kwargs):
        
        return self._add_metriclines_to_single_detection(cc_id, **kwargs)

    def _add_metriclines_to_single_detection(self, 
                                             cc_id, 
                    
                    padding = 30,
                    **kwargs):
        
        import copy
        #print(self._frames_colors)
        if self._frames_colors is None:
            self._frames_colors = random_colors(len(self.bbs))
            
        col = self._frames_colors[cc_id]
        
        imageres = self._img.copy()
        if imageres.shape[0] == 3:
            imageres = imageres.swapaxes(0,1).swapaxes(1,2)
            
        imgclipped = copy.deepcopy(self._clip_image(imageres, self.bbs[cc_id], 
                                                    padding = padding,paddingwithzeros = False))
        maskimage = copy.deepcopy(self._clip_image(self.msks[cc_id], 
                                                               self.bbs[cc_id], 
                                                               padding = padding,
                                                               paddingwithzeros = False))
        
        m = plot_individual_mask(imgclipped, maskimage, textlabel = str(cc_id),
                         col = col, **kwargs)
            
        return m,maskimage
    
    def evaluate_predictions_by_users(self):
        from matplotlib.pyplot import plot, ion, show, close
        
        print("""
              For the current image {} bounding boxes were detected
              which method will you prefer to select the correct predictions\n
              1. by threshhold scores
              2. one by one
              
              """.format(
                         len(self.predictions[0]['scores'])))
        
        evaluation_method = input()       
        imgtoplot = self._imgastensor.mul(255).permute(1, 2, 0).byte().numpy()
        onlythesepos = []
        if evaluation_method == "1":
            minthreshold = input("the mininmum score (0-1): ")
            maxthreshold = input("the maximum score (0-1): ")
            input_1 = float(minthreshold)
            input_2 = float(maxthreshold)
            
            onlythesepos = np.where(np.logical_and(
                self.predictions[0]['scores'].to('cpu').detach().numpy()>input_1,
                self.predictions[0]['scores'].to('cpu').detach().numpy()<input_2))

        if evaluation_method == "2":
            onlythesepos = []
            
            for i in range(len(self.predictions[0]['scores'])):
                mks = self.predictions[0]['masks'].mul(255).byte().cpu().numpy()[i, 0].squeeze()
                bb = self.predictions[0]['boxes'].cpu().detach().numpy()[i]
                
                
                ion()
                f = plot_segmenimages((imgtoplot).astype(np.uint8),
                                      mks, 
                        boxes=[bb.astype(np.uint16)], 
                        bbtype = 'xminyminxmaxymax')
                response = input("is this prediction [{} of {}] correct (1-yes ; 2-no; 3-exit): ".format(
                    i+1,len(self.predictions[0]['scores'])))
                f.show()
                close()
                if response == "1":
                    onlythesepos.append(i)
                if response == "3":
                    break
            
        print(f"in total {len(self.predictions[0]['boxes'].cpu().detach().numpy()[onlythesepos])} segmentations are correct")
        #print()
        
        msks = np.zeros((imgtoplot).astype(np.uint8).shape[:2])
        msks_preds = self.predictions[0]['masks'].mul(255).byte().cpu().numpy()[onlythesepos, 0].squeeze()
        bbs_preds = self.predictions[0]['boxes'].cpu().detach().numpy()[onlythesepos]
        
        if len(msks_preds.shape)>2:
            if msks_preds.shape[0] == 0:
                msks = np.zeros(self.inputsize)
            else:
                msks = np.max(msks_preds, axis = 0)
        
        ion()
        f = plot_segmenimages((imgtoplot).astype(np.uint8),msks, 
                        boxes=bbs_preds.astype(np.uint16), 
                        bbtype = 'xminyminxmaxymax')
        response = input("is this prediction correct (1-yes ; 2-no): ")
        f.show()
        close()
        if response == "2":
            trueresponse= True
            
            while trueresponse:
                response = input("do you want to repeat (1-yes ; 2-no): ")
                if response == "1":
                    trueresponse = False
                    self.evaluate_predictions_by_users()
                elif response == "2":
                    trueresponse = False
                    msks_preds = np.zeros((imgtoplot).astype(np.uint8).shape[:2])
                    bbs_preds = []
                    self.msks = msks_preds
                    self.bbs = bbs_preds
                print("The option is not valid, try again")
                
        else:
            if len(msks_preds.shape) == 2:
                msks_preds = np.expand_dims(msks_preds, axis=0)
            self.msks = msks_preds
            self.bbs = bbs_preds
        
        imgsize = self._img.copy().shape[1:]
            
        #self._img = img
        
        #for i in range(len(self.msks)):
        #    self.msks[i][self.msks[i]<segment_threshold] = 0
        
        self._original_size(imgsize)
            
        return {'masks': self.msks, 
                'bbs':self.bbs, 
                'scores':self.scores}
        #return msks_preds, bbs_preds, imgtoplot
        
    @staticmethod
    def _get_heights_and_widths(maskcontours):

        return get_heights_and_widths(maskcontours)
    
    @staticmethod 
    def _clip_image(image, bounding_box, bbtype = 'xminyminxmaxymax', padding = None, 
                    paddingwithzeros =True):
        
        return clip_image(image, bounding_box, 
               bbtype = bbtype, 
               padding = padding, 
                    paddingwithzeros =paddingwithzeros)

    @staticmethod 
    def _find_contours(image, hull = False):
        
        return find_contours(image, hull = hull)



class SegmentationPrediction():

    
    @check_output_fn
    def load_weights(self, path, fn, suffix = 'pth.tar'):
        
        checkpoint = torch.load(fn, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.opt.load_state_dict(checkpoint["optimizer"])
        print("weights loaded")
    
    def _original_size(self):
        msksc = [0]* len(list(self.msks))
        
        for i in range(len(self.msks)):
            msksc[i] = cv2.resize(self.msks[i][0], 
                                  [self.img.shape[2],self.img.shape[1]], 
                                  interpolation = cv2.INTER_AREA)  
       
        self.msks = np.array(np.expand_dims(msksc, axis =0))
        
    def get_mask(self, image, keepdims = True):
        import collections
        
        self.imgtensor = image_to_tensor(image=image, outputsize = self.inputimgsize)
        self.img = image
        
        self.model.eval()
        with torch.no_grad():
            msks = self.model(torch.stack([self.imgtensor.to(self.device)]))
            
        if type(msks) is collections.OrderedDict:
            #self.msks = self.msks['out']
            self.msks =msks['out'].mul(255).byte().cpu().numpy()
        else:
            self.msks = msks.mul(255).byte().cpu().numpy()
        
        if keepdims:
            self._original_size()
        
        return self.msks

    def set_model(self):
        
        if self.arch == "Unet256":
            model = Unet256(in_channels=3, out_channels=1)
        
        return model.to(self.device)
    
    def __init__(self, architecture = "Unet256", configuration = {'lr':2e-4,
                                                                  'beta': (0.5, 0.999)}, device = None) -> None:
        
        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.arch = architecture
        self.model = self.set_model()
        self.inputimgsize = (256, 256)
        self.opt = optim.Adam(self.model.parameters(), 
                      lr=configuration['lr'], betas=configuration['beta'])