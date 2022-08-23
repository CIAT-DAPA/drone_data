import math

import concurrent.futures as cf
import geopandas as gpd
import numpy as np
import pandas as pd

import linecache
import xarray
import os

from scipy.stats import gaussian_kde


from utils.classification_functions import kmeans_images
from utils.gis_functions import transform_frombb, rasterize_using_bb,list_tif_2xarray
from utils.plt_functions import plot_2d_cloudpoints
from sklearn.neighbors import KNeighborsRegressor
from pykrige.ok import OrdinaryKriging


def getchunksize_forxyzfile(file_path, bb,buffer, step = 100):

    cond1 = True
    idx = 0
    idx2 = step
    while cond1:
        try:
            fl = linecache.getline(file_path,idx2).split(' ')[0]
            cond2 = float(fl)>=(bb[0] - buffer)
            if cond2:
                cond1 =float(fl)<=(bb[2] + buffer)
            else:
                idx = idx2
            
            idx2+=step
        except:
            idx2 = 0
            cond1 = False
    if idx != 0:
        idxdif = idx2 - idx
    else:
        idxdif=0

    if np.abs(idxdif) <2000:
        idxdif=0

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

def read_cloudpointsfromxyz(file_path, bb, buffer= 0.1, step = 1000, ext='.xyz',mindata = 100):
    data = True
    file_pathfn = os.listdir(file_path)
    xyzfilenames = [i for i in file_pathfn if i.endswith(ext)]
    
    count = 0
    while data:
        firstrow,chunksize = getchunksize_forxyzfile(
            os.path.join(file_path,xyzfilenames[count]), bb,buffer, step)
        if chunksize>0:
            chunks = pd.read_csv(
                os.path.join(file_path,xyzfilenames[count]),
                skiprows=firstrow, chunksize=chunksize, header=None, sep = " ")

            if count == 0:
                df = pd.concat(valid(chunks, bb, buffer))
                dfp =df.copy()
                if len(dfp)>mindata:
                    data = False
            else:
                df = pd.concat(valid(chunks, bb, buffer))
                dfp = pd.concat([dfp,df])
        
        if count >=(len(xyzfilenames)-1):
           data = False
        count +=1

    if len(dfp)<mindata:
        raise ValueError('Check the coordinates, there is no intesection in the file')

    return dfp


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


#def interpolate_cloud_points(
#    dfcloudlist, image_shape,
#)

def from_cloudpoints_to_xarray(dfcloudlist, 
                               bounds,
                               coords_system,
                               columns_name = ["z", "red","green", "blue"],
                               spatial_res = 0.01,
                               dimension_name= "date",
                               newdim_values = None,
                               interpolate = False,
                               inter_method = 'KNN', **kargs):

    trans, imgsize = transform_frombb(bounds, spatial_res)
    totallength = len(columns_name)+2
    xarraylist = []
    for j in range(len(dfcloudlist)):
        list_rasters = []
        xycoords = dfcloudlist[0][[1,0]].values.copy()

        for i in range(2,totallength):
            valuestorasterize = dfcloudlist[j].iloc[:,[i]].iloc[:, 0].values
            rasterinterpolated = rasterize_using_bb(valuestorasterize, 
                                                    dfcloudlist[j].geometry, 
                                                    transform = trans, imgsize =imgsize)

            
            if interpolate:
                
                rasterinterpolated = points_rasterinterpolated(
                    (xycoords.T[0],xycoords.T[1],valuestorasterize), 
                    transform = trans, 
                    rastershape = rasterinterpolated.shape,
                    inter_method= inter_method,
                    **kargs)

     
            list_rasters.append(rasterinterpolated)

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


def calculate_leaf_angle(xrdata, vector = (0,0,1), invert = False,heightvarname = 'z', name4d ='date'):
    
    varnames = list(xrdata.keys())
    if heightvarname is not None and heightvarname not in varnames:
        raise ValueError('{} is not in the xarray'.format(heightvarname))

    anglelist = []
    #name4d = list(xrdata.dims.keys())[0]
    if len(xrdata.dims.keys())>2:
        for dateoi in range(len(xrdata[name4d])):
            anglelist.append(get_angle_image_fromxarray(
                xrdata.isel({name4d:dateoi}).copy(), vcenter=vector,heightvarname = heightvarname))
        
        xrimg = xarray.DataArray(anglelist)
        vars = list(xrdata.dims.keys())
        
        vars = [vars[i] for i in range(len(vars)) if i != vars.index(name4d)]

        xrimg.name = "leaf_angle"
        xrimg = xrimg.rename({'dim_0': name4d, 
                            'dim_1': vars[0],
                            'dim_2': vars[1]})
    else:
        xrimg = xarray.DataArray(get_angle_image_fromxarray(
                xrdata.copy(), vcenter=vector,heightvarname = heightvarname))
        vars = list(xrdata.dims.keys())
        xrimg.name = "leaf_angle"
        xrimg = xrimg.rename({'dim_0': vars[0], 
                            'dim_1': vars[1]})

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

def clip_cloudpoints_as_gpd(file_name, bb, crs, buffer = 0.1, step = 100, ext = '.xyz'):
    

    dfcl = read_cloudpointsfromxyz(file_name,  
                            bb.bounds, 
                            buffer = buffer,step = step,
                            ext =ext)


    dfcl = gpd.GeoDataFrame(dfcl, geometry=gpd.points_from_xy(
                                    dfcl.iloc[:,0], dfcl.iloc[:,1]), crs=crs)

    return dfcl.loc[dfcl.within(bb)]

def points_to_raster_interp(points, grid, method = "KNN", 
                            knn = 5, weights = "distance",
                            variogram_model = 'hole-effect'):
    """
    this function interpolates points where the surlt is an image

    Parameters:
    ----------
    points: list
        this a list that contains three list, points in x, points in y and the 
        values to be interpolated
    grid: list
        a list that contains the meshgrids in x and y.
    method: str, optional
        a string that describes which interpolated method will be used, 
        currently only KNN and ordinary_kriging are available

    
    Parameters:
    ----------
    interpolated image

    """
    if len(grid) == 2:
        xx = grid[0]
        yy = grid[1]
    else:
        raise ValueError("Meshgrid must have values for x and y")
    if len(points) == 3:
        coordsx = points[0]
        coordsy = points[1]
        values = points[2]
    else:
        raise ValueError("Points is a list that has three lists, one ofr x, y and Zvalues")

    if method == "KNN":
        regressor = KNeighborsRegressor(n_neighbors = knn, weights = weights)
        regressor.fit(list(zip(coordsx,coordsy)), values)
            ## prediction
        imgpredicted = regressor.predict(
            np.array((xx.ravel(),yy.ravel())).T).reshape(
                xx.shape)

    if method == "ordinary_kriging":
        ok = OrdinaryKriging(
            coordsx,
            coordsy,
            values,
            variogram_model = variogram_model,
        )
        ## prediction
        imgpredicted, _ = ok.execute("grid", 
                np.unique(xx.ravel()),np.unique(yy.ravel()))
        del ok
        imgpredicted = imgpredicted.swapaxes(0,1)

    return imgpredicted


def points_rasterinterpolated(points, transform, rastershape, inter_method = 'KNN', **kargs):
        from utils.gis_functions import coordinates_fromtransform
        x, y = coordinates_fromtransform(transform,
                            [rastershape[1], rastershape[0]])

        xx,yy = np.meshgrid(np.sort(np.unique(x)), np.sort(np.unique(y)))

        rastinterp = points_to_raster_interp(
                            points,
                            (yy,xx), method = inter_method, **kargs)

        return rastinterp
                


class CloudPoints:

    def to_xarray(self, sp_res = 0.01, newdim_values = None, interpolate = False, inter_method = "KNN"):
        """
        This function will create a spatial raster with the cloud points
        the spatial image can be obtained by rasterizing the vector points, or applying 
        a spatial interpolation 

        Parameters:
        ----------
        sp_res: float, optional
            final spatial resolution used for vector rasterization
        newdim_values: str, optional
            reasign new names for the xarray dimensions
        interpolate: str, optional
            which method will be applied to get the spatial image 
            ['rasterize', 'interpolation'], default rasterize

        """
        
        rasterimg = from_cloudpoints_to_xarray(self.cloud_points,
                                   self.boundaries, 
                                   self._crs,
                                   self.variables_names,
                                   spatial_res = sp_res,
                                   newdim_values = newdim_values,
                                   interpolate = interpolate,
                                   inter_method = inter_method)
                                   
        #elif method == 'interpolation':
        ### TODO: interpolation metod knn 
        """
        from sklearn.neighbors import KNeighborsRegressor
        zvalues = (points_perplant.cloud_points[0][2].values.copy()-bsl)*100
        xycoords = points_perplant.cloud_points[0][[1,0]].values.copy()
        knn_regressor = KNeighborsRegressor(n_neighbors = 7, weights = "distance")
        knn_regressor.fit(xycoords, zvalues)

        """
        return rasterimg

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
                    verbose = False,
                    ext = '.xyz'):

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

