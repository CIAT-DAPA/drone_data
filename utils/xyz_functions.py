import math

import concurrent.futures as cf
import geopandas as gpd
import numpy as np
import pandas as pd

import linecache
import xarray

from scipy.stats import gaussian_kde


from utils.classification_functions import kmeans_images
from utils.gis_functions import transform_frombb, rasterize_using_bb,list_tif_2xarray
from utils.plt_functions import plot_2d_cloudpoints


def getchunksize_forxyzfile(file_path, bb,buffer, step = 100):

    cond1 = True
    idx2 = step
    while cond1:
        cond2 = float(linecache.getline(file_path,idx2).split(' ')[0])>=(bb[0] - buffer)
        if cond2:
            cond1 =float(linecache.getline(file_path,idx2).split(' ')[0])<=(bb[2] + buffer)
        else:
            idx = idx2
        
        idx2+=step
    idxdif = idx2 - idx
    linecache.clearcache()
    return ([idx, idxdif])



def valid(chunks, bb, buffer= 0.0):

    for chunk in chunks:
        mask = np.logical_and(
            (np.logical_and((chunk.iloc[:,1] > (bb[1]-buffer)) ,(chunk.iloc[:,1] < (bb[3]+buffer)))),
            (np.logical_and((chunk.iloc[:,0] > (bb[0]-buffer)), (chunk.iloc[:,0] < (bb[2]+buffer)))))
        if mask.all():
            yield chunk
        else:
            yield chunk.loc[mask]
            break

def read_cloudpointsfromxyz(file_path, bb, buffer= 0.1, step = 1000):

    firstrow,chunksize = getchunksize_forxyzfile(file_path, bb,buffer, step)
    
    chunks = pd.read_csv(file_path,skiprows=firstrow, chunksize=chunksize, header=None, sep = " ")
    df = pd.concat(valid(chunks, bb, buffer))
    
    return df


def get_baseline_altitude(clouddf, nclusters = 15, nmaxcl = 4, method = 'max_probability', 
                          quantile_val = .85, stdtimes = 1):


    df = clouddf.copy()
    bsl = None
    if method == 'cluster':
        clust = kmeans_images(df, nclusters)
        df = df.assign(cluster = clust['labels'])

        bsl = df.groupby('cluster').agg({2: 'mean'}
            ).sort_values(by=[2], ascending=False).iloc[0:nmaxcl].mean().values[0]

    if method == 'max_probability':

        ydata = df.iloc[:,1].values.copy()
        zdata = df.iloc[:,2].values.copy()
        ycentermask1 = ydata>(np.mean(ydata)+(stdtimes*np.std(ydata)))
        ycentermask2 = ydata<(np.mean(ydata)-(stdtimes*np.std(ydata)))
        datam = np.sort(zdata[ycentermask1])
        datah = np.sort(zdata[ycentermask2])
        ys1 = gaussian_kde(datam)
        ys2 = gaussian_kde(datah)
        valmax1 = datam[np.argmax(ys1(datam))]
        valmax2 = datah[np.argmax(ys2(datah))]
        bsl = (valmax1 + valmax2)/2

    if method == "quantile":
        bsl = df.iloc[:,2].quantile(quantile_val)
    
    return bsl

def from_cloudpoints_to_xarray(dfcloudlist, 
                               bounds,
                               coords_system,
                               columns_name = ["z", "red","green", "blue"],
                               spatial_res = 0.01,
                               dimension_name= "date",
                               newdim_values = None):

    trans, _ = transform_frombb(bounds, spatial_res)
    totallength = len(columns_name)+2
    xarraylist = []
    for j in range(len(dfcloudlist)):
        list_rasters = []
        for i in range(2,totallength):
            list_rasters.append(rasterize_using_bb(dfcloudlist[j].iloc[:,[i,totallength]], 
                                                   bounds, crs = coords_system, sres = spatial_res))

        xarraylist.append(list_tif_2xarray(list_rasters, trans, 
                                           crs = coords_system,
                                           bands_names = columns_name))

    mltxarray = xarray.concat(xarraylist, dim=dimension_name)
    mltxarray.assign_coords(date = [m+1 for m in range(len(dfcloudlist))])
    if newdim_values is not None:
        if len(newdim_values) == len(dfcloudlist):
            mltxarray[dimension_name] = newdim_values
        else:
            print("dimension and names length does not match")

    return mltxarray


def calculate_leaf_angle(xrdata, vector = (0,0,1), invert = False,heightvarname = 'z'):
    
    varnames = list(xrdata.keys())
    if heightvarname is not None and heightvarname not in varnames:
        raise ValueError('{} is not in the xarray'.format(heightvarname))

    anglelist = []
    name4d = list(xrdata.dims.keys())[0]

    for dateoi in range(len(xrdata[name4d])):
        anglelist.append(get_angle_image_fromxarray(
            xrdata.isel({name4d:dateoi}).copy(), vcenter=vector,heightvarname = heightvarname))
    
    xrimg = xarray.DataArray(anglelist)
    xrimg.name = "leaf_angle"
    xrimg = xrimg.rename({'dim_0': list(xrdata.dims.keys())[0], 
                          'dim_1': list(xrdata.dims.keys())[1],
                          'dim_2': list(xrdata.dims.keys())[2]})
    
    xrdata = xrdata.merge(xrimg)
    
    if invert:
        xrdata["leaf_angle"] = 90-xrdata["leaf_angle"]
    
    return xrdata



def get_angle_image_fromxarray(xrdata, vcenter = (1,1,0),heightvarname = 'z'):

    df = xrdata.to_dataframe().copy()
    
    ycoords = np.array([float("{}.{}".format(str(i[0]).split('.')[0][-3:], str(i[0]).split('.')[1])) for i in df.index.values])*100
    xcoords = np.array([float("{}.{}".format(str(i[1]).split('.')[0][-3:], str(i[1]).split('.')[1]))  for i in df.index.values])*100
    
    xcenter = np.mean(xcoords)
    ycenter = np.mean(ycoords)

    anglelist = []
    for x,y,z in zip(xcoords,ycoords,xrdata[heightvarname].values.ravel()):
        anglelist.append(math.degrees(calculate_angle_twovectors(vcenter , ((x-xcenter), (y-ycenter),z))))

    
    anglelist = np.array(anglelist).reshape(xrdata[heightvarname].shape)
    #anglelist[xrdata[heightvarname].values == 0] = 0
    return(anglelist)


def remove_bsl_toxarray(xarraydata, baselineval, scale_height = 100):
                
    xrfiltered = xarraydata.where(xarraydata.z > baselineval, np.nan)
    xrfiltered.z.loc[:] = (xrfiltered.z.loc[:] - baselineval)*scale_height

    return xrfiltered
    

def calculate_angle_twovectors(v1,v2):
    
    dot_product = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return( np.arccos(dot_product))

def clip_cloudpoints_as_gpd(file_name, bb, crs, buffer = 0.1, step = 100):
    
    dfcl = read_cloudpointsfromxyz(file_name,  
                            bb.bounds, 
                            buffer = buffer,step = step)

    dfcl = gpd.GeoDataFrame(dfcl, geometry=gpd.points_from_xy(
                                    dfcl.iloc[:,0], dfcl.iloc[:,1]), crs=crs)

    return dfcl.loc[dfcl.within(bb)]


class CloudPoints:

    def to_xarray(self, sp_res = 0.01, newdim_values = None):
        return from_cloudpoints_to_xarray(self.cloud_points,
                                   self.boundaries, 
                                   self._crs,
                                   self.variables_names,
                                   spatial_res = sp_res,
                                   newdim_values = newdim_values)


    def remove_baseline(self, method= None, 
                        cloud_reference = 0, scale_height = 100, 
                        applybsl = True,baselineval = None,**kargs):
        if method is None:
            method = "max_probability"
        
        if baselineval is None:
            bsl = get_baseline_altitude(self.cloud_points[cloud_reference].iloc[:,0:6], method=method , **kargs)
        else:
            bsl = baselineval

        self._bsl = bsl
        #print("the baseline used was {}".format(bsl))
        if applybsl:
            for i in range(len(self.cloud_points)):
                data = self.cloud_points[i].copy()
                data = data.loc[data.iloc[:,2]>=bsl,:]
                data.iloc[:,2] = (data.iloc[:,2].values-bsl)*scale_height

                self.cloud_points[i] = data

    def plot_2d_cloudpoints(self, index = 0, figsize = (10,6), xaxis = "latitude"):
        return plot_2d_cloudpoints(self.cloud_points[index], figsize, xaxis)

    def __init__(self, 
                    xyzfile,
                    gpdpolygon,
                    buffer = 0.1,
                    step = 1000,
                    crs = 32654,
                    variables = ["z", "red","green", "blue"],
                    asxarray = False,
                    sp_res = 0.01,
                    multiprocess = False,
                    nworkers = 2, 
                    verbose = False):

        self._crs =  crs     
        self.boundaries = gpdpolygon.bounds
        self.variables_names = variables


        if type(xyzfile) != list:
            xyzfile = [xyzfile]


        cllist = []
        if multiprocess:
            #print("Multiprocess initialization")
            cloud_thread = []
            with cf.ProcessPoolExecutor(max_workers=nworkers) as executor:
                for i in range(len(xyzfile)):
                    cloud_thread.append({executor.submit(clip_cloudpoints_as_gpd,xyzfile[i], 
                                            gpdpolygon,
                                            self._crs,
                                            buffer,
                                            step): i})

            cllist = []
            for future in cloud_thread:
                for i in cf.as_completed(future):
                    cllist.append(i.result())
 
           
        else:
            for i in range(len(xyzfile)):
                if verbose:
                    print(xyzfile[i])
    
                gdf =  clip_cloudpoints_as_gpd(xyzfile[i],gpdpolygon, 
                                               crs=self._crs,
                                               buffer = buffer,
                                               step = step)
                if asxarray:
                    gdf = from_cloudpoints_to_xarray([gdf],
                                   self.boundaries, 
                                   self._crs,
                                   self.variables_names,
                                   spatial_res = sp_res,
                                   newdim_values = 'date')
                cllist.append(gdf)
            
            if asxarray:
                lenimgs = len(cllist)
                cllist = xarray.concat(cllist, dim='date')
                cllist.assign_coords(date = [m+1 for m in range(lenimgs)])

        self.cloud_points = cllist


