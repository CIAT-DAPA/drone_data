from .utils import xyxy_to_xywh, from_yolo_toxy
from .plt_utils import draw_frame
from ..utils.drone_data import DroneData
from ..utils.gis_functions import from_polygon_2bbox
from ..utils.general import find_postinlist

import geopandas as gpd


class UAVBB_fromvector(DroneData):
    """Class to create labels ferom spatial vector files. Currently only process one polygon at the time

    Args:
        DroneData (class): Class created for processing uav imagery and keep them as xarray

    Returns:
        txt: yolo file
    """
    @property
    def xycoords(self):
        x1, y1, x2, y2 = from_polygon_2bbox(self.geom_polygon)
        return x1, y1, x2, y2
    
    @property
    def geom_polygon(self):
        assert type(self.boundary) is gpd.GeoDataFrame
        geompolygon = self.boundary.geometry.values[0]
        return geompolygon
    
    @property
    def image_coords(self):
        
        xcoords = self.drone_data.coords['x'].values.copy()
        ycoords = self.drone_data.coords['y'].values.copy()
        
        l = find_postinlist(xcoords, self.xycoords[0])
        b = find_postinlist(ycoords, self.xycoords[1])
        r = find_postinlist(xcoords, self.xycoords[2])
        t = find_postinlist(ycoords, self.xycoords[3])
        
        return [l,b,r,t]
    
    @property
    def xyhw_coords(self):
        return xyxy_to_xywh(self.image_coords)
    
    def yolo_style(self, labelid = None):
        imc= self.drone_data.copy().to_array().values.swapaxes(0,1).swapaxes(1,2)
        x,y,h,w = self.xyhw_coords

        labelid = 0 if labelid is None else labelid
        
        heigth = imc.shape[1]
        width = imc.shape[0]

        return [0, x/width,y/heigth,h/heigth,w/width]
            
    def bb_plot(self, labelid = None):
        """Draw a plot with the image

        Args:
            labelid (str): category's name. Defaults to None.

        Returns:
            numpy.array: image
        """
        labelid = 0 if labelid is None else labelid
        imc= self.drone_data.copy().to_array().values.swapaxes(0,1).swapaxes(1,2)
        
        imgdr = draw_frame(imc.copy(), 
                    [from_yolo_toxy(self.yolo_style(labelid), imc.shape[:2])])
        
        return imgdr

    def __init__(self, inputpath, bands=None, multiband_image=False, bounds=None, buffer = None):
        
        self.boundary = bounds.copy()
        
        if buffer is not None:
            boundary = bounds.copy().buffer(buffer, join_style=2)
        else:
            boundary = bounds.copy()
            
        super().__init__(inputpath, bands, multiband_image, boundary)
        