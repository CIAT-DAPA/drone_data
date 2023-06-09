import math

import concurrent.futures as cf
import geopandas as gpd
import numpy as np
import pandas as pd

import linecache
import xarray
import os

from scipy.stats import gaussian_kde

from .classification_functions import kmeans_images
from .gis_functions import transform_frombb, rasterize_using_bb,list_tif_2xarray
from .plt_functions import plot_2d_cloudpoints

from sklearn.neighbors import KNeighborsRegressor
from pykrige.ok import OrdinaryKriging



def getchunksize_forxyzfile(file_path, bb,buffer, step = 100):

    cond1 = True
    idinit = 0
    idx2 = step
    
    idxdif= 0
    #this is a flag to point out that the file doesnt have all information
    initfinalpos = [0,0]
    while cond1:
        try:
            firstvalueintherow = linecache.getline(file_path,idx2).split(' ')[0]
            cond2 = float(firstvalueintherow)>=(bb[0] - buffer)
            if cond2:
                initfinalpos = [idinit,idx2]
                cond1 =float(firstvalueintherow)<=(bb[2] + buffer)
            else:
                idinit = idx2
            
            idx2+=step
        except:
            break
    
    #if idinit != 0:
    idxdif = initfinalpos[1] - initfinalpos[0]

    if idxdif < 2000:
        idxdif=0

    linecache.clearcache()
    return ([initfinalpos[0], idxdif])


def filterdata_insidebounds(chunks, bb, buffer= 0.0):

    for chunk in chunks:
        mask = np.logical_and(
            (np.logical_and((chunk.iloc[:,1] > (bb[1]-buffer)) ,(chunk.iloc[:,1] < (bb[3]+buffer)))),
            (np.logical_and((chunk.iloc[:,0] > (bb[0]-buffer)), (chunk.iloc[:,0] < (bb[2]+buffer)))))
        if mask.all():
            yield chunk
        else:
            yield chunk.loc[mask]
            break


def read_cloudpointsfromxyz(file_path, bb, buffer= 0.1, sp_res = 0.005, ext='.xyz'):
    """
    a function to read a cloud points file using spatial boundaries

    Parameters:
    ----------
    file_path: str
        3d point cloud file location
    bb: tuple
        spatial bounding box
    sp_res: float, optional
        the camera spatial resolution in meters, default 0.005
    ext: str, optional
        which is the file extension, default '.xyz'

    Return:
    ----------
    Pandas dataframe with columns (x, y , z , r, g , b)

    """
    width = abs(bb[0]-bb[2]) + buffer
    heigth = abs(bb[1]-bb[3]) + buffer
    mindata = int(heigth/sp_res * width/sp_res)
    ### junp every step onf lines in the point cloud file
    step = int(mindata*0.015)

    undermindata = True
    if file_path.endswith('.xyz'):
        folders = os.path.split(file_path)
        file_path = folders[0]
        xyzfilenames = [folders[-1]]
    else:
        file_pathfn = os.listdir(file_path)
        xyzfilenames = [i for i in file_pathfn if i.endswith(ext)]
    
    count = 0
    dfp = []
    sizefiles = 0
    while undermindata:

        firstrow,chunksize = getchunksize_forxyzfile(
            os.path.join(file_path,xyzfilenames[count]), bb,buffer, step)
        
        if chunksize>0:
            chunks = pd.read_csv(
                os.path.join(file_path,xyzfilenames[count]),
                skiprows=firstrow, chunksize=chunksize, header=None, sep = " ")
            
            df = pd.concat(filterdata_insidebounds(chunks, bb, buffer))
            dfp.append(df)
            sizefiles = len(df) + sizefiles
        
        if count >=(len(xyzfilenames)-1) or sizefiles>mindata:
           undermindata = False
        count +=1
    
    if sizefiles<mindata:
        raise ValueError('Check the coordinates, there is no intesection in the file')
    else:
        dfp = pd.concat(dfp)

    return dfp


def get_baseline_altitude(clouddf, nclusters = 15, nmaxcl = 4, method = 'max_probability', 
                          quantile_val = .85, stdtimes = 1):
    
    """
    a function to get the distribution value for a given quantile, for both border
    ...
    Parameters
    ----------
    nclusters:
        if cluster method is activated, the number of cluster which the data will be clustered
    nmaxcl:
        number of cluter which will be used to average and obtain a baseline
    """


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
        zeromask = np.logical_not(zdata==0)
        ydata = ydata[zeromask]
        zdata = zdata[zeromask]
        
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

def from_cloudpoints_to_xarray(dfpointcloud, 
                               bounds,
                               coords_system,
                               columns_name = ["z", "red","green", "blue"],
                               spatial_res = 0.01,
                               dimension_name= "date",
                               newdim_values = None,
                               interpolate = False,
                               inter_method = 'KNN',
                               knn = 5, weights = "distance",
                               variogram_model = 'exponential',
                               ):

    """
    the point cloud is a 3D representated as points with coordinates in xand y, and with values in a third axis z. This function
    transform the 3D representation to a 2D image. To obtain this, a rasterize processes is applied. The rasterization can be 
    obtained from a vextorization or by spatial interpolation.

    Parameters:
    ----------
    dfpointcloud: list
        a list that contains all the oint cloud data frames to be processed
    bounds: polygon
        this is a geopandas geometry that will be used boundaries
    coords_system: str
        Coordinates system reference
    spatial_res: float
        Spatial reolution that the image will have in meters, default 0.01
    interpolate: boolean
        If it is True, an interpolation method will be applied, if not a rasterize (averagering points that intersect a cell) method is gonna be held.
    
    Return:
    ----------
    xarray

    """
    trans, imgsize = transform_frombb(bounds, spatial_res)
    totallength = len(columns_name)+2
    xarraylist = []
    for j in range(len(dfpointcloud)):
        list_rasters = []
        xycoords = dfpointcloud[j][[0,1]].values.copy()

        for i in range(2,totallength):
            valuestorasterize = dfpointcloud[j].iloc[:,[i]].iloc[:, 0].values
            
            if interpolate:
                rasterinterpolated = points_rasterinterpolated(
                    (xycoords.T[0],xycoords.T[1],valuestorasterize), 
                    transform = trans, 
                    rastershape = imgsize,
                    inter_method= inter_method,
                               knn = knn, weights = weights,
                               variogram_model = variogram_model)
            else:
                rasterinterpolated = rasterize_using_bb(valuestorasterize, 
                                                    dfpointcloud[j].geometry, 
                                                    transform = trans, imgsize =imgsize)

            
            

     
            list_rasters.append(rasterinterpolated)

        xarraylist.append(list_tif_2xarray(list_rasters, trans, 
                                           crs = coords_system,
                                           bands_names = columns_name))

    if len(xarraylist) > 1:
        mltxarray = xarray.concat(xarraylist, dim=dimension_name)
        mltxarray.assign_coords(date = [m+1 for m in range(len(dfpointcloud))])
    else:
        mltxarray = xarraylist[0]

    if newdim_values is not None:
        if len(newdim_values) == len(dfpointcloud):
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

def clip_cloudpoints_as_gpd(file_name, bb, crs, buffer = 0.1, sp_res = 0.005, ext = '.xyz'):
    

    dfcl = read_cloudpointsfromxyz(file_name,  
                            bb.bounds, 
                            buffer = buffer,
                            sp_res = sp_res,
                            ext =ext)


    dfcl = gpd.GeoDataFrame(dfcl, geometry=gpd.points_from_xy(
                                    dfcl.iloc[:,0], dfcl.iloc[:,1]), crs=crs)

    return dfcl.loc[dfcl.within(bb)]

def points_to_raster_interp(points, grid, method = "KNN", 
                            knn = 5, weights = "distance",
                            variogram_model = 'hole-effect'):
    """
    this function comput a spatial interpolation using 

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
    variogram_model: str, optional
        linear, exponential, power, hole-effect
    
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
                
    #https://mmaelicke.github.io/scikit-gstat/_modules/skgstat/Kriging.html
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
    """_summary_

    Args:
        points (pandas.DataFrame): point cloud dataframe
        transform (Affine): raster transformation matrix 
        rastershape (list): image size (Height x Width)
        inter_method (str, optional): _description_. Defaults to 'KNN'.

    Returns:
        numpy array: interpolated image
    """
    from drone_data.utils.gis_functions import coordinates_fromtransform
    rows, columns = coordinates_fromtransform(transform,
                        [rastershape[0], rastershape[1]])

    yy,xx = np.meshgrid(np.sort(np.unique(columns)), np.sort(np.unique(rows)))
    
    rastinterp = points_to_raster_interp(
                        points,
                        (yy,xx), method = inter_method, **kargs)

    return rastinterp
                


class CloudPoints:
    """
    A class used to process XYZ files, this reads the file and then based on a boundary vector file
    returns shrink the clou points to that region only. 

    ...

    Attributes
    ----------
    boundaries : geopandas geometry
        a formatted string to print out what the animal says
    variables_names : list
        the name of the features that are in the XYZ file
    cloud_points : pandas
        a dataframe table that contains the cloud points for an espectific bondary.

    Methods
    -------
    to_xarray(sp_res=float)
        transform the cloud points file to a geospatial raster
    """


    @property
    def cloud_points(self):
        return self._point_cloud

    def _cloud_point(self):
        cllist = []

        for i in range(len(self.xyzfile)):
            if self.verbose:
                print(self.xyzfile[i])

            gdf =  clip_cloudpoints_as_gpd(self.xyzfile[i],self.geometry, 
                                            crs=self._crs,
                                            buffer = self.buffer,
                                            sp_res = self.spatial_res)

            cllist.append(gdf)
            
        self._point_cloud =  cllist   


    def to_xarray(self, sp_res = 0.01, 
                  newdim_values = None, 
                  interpolate = False, 
                  inter_method = "KNN",
                  **kargs):
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
        interpolate: boolean, optional
            which method will be applied to get the spatial image 
            ['rasterize', 'interpolation'], default rasterize
        inter_method: str, optional
            Set the interpolation method, currently there are available
             k-nearest neighbors 'KNN" and Ordinary-Kriging 'ordinary_kriging'
        
        Returns:
        ----------
        xarray file that contains the raster image

        """
        
        self.twod_image = from_cloudpoints_to_xarray(self.cloud_points,
                                   self.geometry.bounds, 
                                   self._crs,
                                   self.variables_names,
                                   spatial_res = sp_res,
                                   newdim_values = newdim_values,
                                   interpolate = interpolate,
                                   inter_method = inter_method,
                                   **kargs)
                                   

        return self.twod_image

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

    def plot_2d_cloudpoints(self, index = 0, figsize = (10,6), xaxis = "latitude",fontsize = 12):
        return plot_2d_cloudpoints(self.cloud_points[index], figsize, xaxis,fontsize=fontsize)

    def __init__(self, 
                    xyzfile,
                    gpdpolygon,
                    buffer = 0.1,
                    spatial_res = 0.01,
                    crs = 32654,
                    variables = ["z", "red","green", "blue"],
                    verbose = False,
                    ext = '.xyz'):

        self._crs =  crs     
        
        self.variables_names = variables

        if type(xyzfile) != list:
            xyzfile = [xyzfile]

        if (type(gpdpolygon) is gpd.GeoSeries) or (type(gpdpolygon) is gpd.GeoDataFrame):
            gpdpolygon = gpdpolygon.reset_index().geometry[0]

        self.boundaries = gpdpolygon.bounds
        self.xyzfile = xyzfile
        self.geometry = gpdpolygon
        self.buffer = buffer
        self.spatial_res = spatial_res
        self.verbose = verbose
        self._xyz_file_suffix = ext
        self._cloud_point()



"""
        if self.multiprocess:
            #print("Multiprocess initialization")
            cloud_thread = []
            with cf.ProcessPoolExecutor(max_workers=self.nworkers) as executor:
                for i in range(len(self.xyzfile)):
                    cloud_thread.append({executor.submit(clip_cloudpoints_as_gpd,self.xyzfile[i], 
                                            self.geometry,
                                            self._crs,
                                            self.buffer,
                                            self.step): i})

            cllist = []
            for future in cloud_thread:
                for i in cf.as_completed(future):
                    cllist.append(i.result())
 
           
        else:
"""