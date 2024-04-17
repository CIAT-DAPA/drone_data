
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage.morphology import convex_hull_image

import os
import numpy as np
import pandas as pd
import xarray
import math
import random

from .datacube_transforms import DataCubeProcessing
from .xr_functions import CustomXarray, from_dict_toxarray, calculate_lab_from_xarray, crop_xarray_using_mask
from .drone_data import calculate_vi_fromxarray
from .gis_functions import centerto_edgedistances_fromxarray,get_filteredimage
from .image_functions import getcenter_from_hull, calculate_differencesinshape, img_padding

from typing import List, Optional, Dict


MORPHOLOGICAL_METRICS = [
    'z', 
    'leaf_angle',
    'volume']

SPECTRAL_METRICS =[
    'edge',
    'nir',
    'ndvi',
    'ndre'
]


EARLYSTAGES_PHENOMIC_METRICS = [
    'leaf_area', 
    'rosette_radius',
    'convex_hull_area']



def tsquantiles_plot(df, yname = None, splitname = 'metric',figsize = (12,6),colormap = 'Dark2', xname = 'date'):

    figure, ax= plt.subplots(figsize = figsize)
    clmapvalues = cm.get_cmap(colormap, len(np.unique(df[splitname])))

    for i, q in enumerate(np.unique(df[splitname])):
        
        ss = df.loc[df[splitname] == q]
        #ax.plot(ss.date, ss.value, label = i)
        ax.plot(ss.date, ss.value, marker = 'o', label = q, c = clmapvalues(i))

    ax.set_xlabel(xname, fontsize=18)
    if yname is not None:
        ax.set_ylabel(yname, fontsize=18)

    ax.legend()

    return ax

def ts_plot(df, yname = None, figsize = (12,6),colormap = 'Dark2', xname = 'date'):
    
    figure, ax= plt.subplots(figsize = figsize)
    clmapvalues = cm.get_cmap(colormap, len(np.unique(1)))

    #ax.plot(ss.date, ss.value, label = i)
    ax.plot(df.date, df.value, marker = 'o', c = clmapvalues(0))
    ax.set_xlabel(xname, fontsize=18)
    if yname is not None:
        ax.set_ylabel(yname, fontsize=18)

    return ax

def get_df_quantiles(xrdata, varname=None, quantiles = [0.25,0.5,0.75]):

    if varname is None:
        varname = list(xrdata.keys())[0]
    
    if varname not in list(xrdata.keys()):
            raise ValueError(
                'the variable name {} is not in the xarray data'.format(varname))
    
    df = xrdata[varname].copy().to_dataframe()
    
    if len(list(xrdata.dims.keys())) >=3:
        
        vardatename = [i for i in list(xrdata.dims.keys()) if type(xrdata[i].values[0]) == np.datetime64]
        
        df = df.groupby(vardatename).quantile(quantiles).unstack()[varname]
        df.columns = ['q_{}'.format(i) for i in quantiles]
        df = df.unstack().reset_index()
        df.columns = ['quantnames', 'date', 'value']
    ## not date include
    else:
        vals = df.quantile(quantiles).unstack()[varname].reset_index().T.iloc[1]
        df = pd.DataFrame(zip(['q_{}'.format(i) for i in quantiles], vals.values))
        df.columns = ['quantnames', 'value']
    #times = ['_t' + str(list(np.unique(df.date)).index(i)) for i in df.date.values]

    df['metric'] = [varname + '_'] + df['quantnames']

    return df.drop(['quantnames'], axis = 1)


def calculate_volume(xrdata, method = 'leaf_angle', heightvarname = 'z', 
                                                    leaf_anglename ='leaf_angle',
                                                    leaf_anglethresh = 70,
                                                    reduction_perc = 40,
                                                    name4d = 'date', wrapper = "hull"):
    


    if heightvarname not in list(xrdata.keys()):
        raise ValueError('the height variable is not in the xarray')
    

    pixelarea =(xrdata.attrs['transform'][0] *100) * (xrdata.attrs['transform'][0] *100)

    if method == 'leaf_angle' and leaf_anglename in list(xrdata.keys()):
        xrfiltered = xrdata.where((xrdata[leaf_anglename])>leaf_anglethresh,np.nan)[heightvarname].copy()
    elif (method == 'window'):
                
        xrfiltered = get_filteredimage(xrdata, channel = heightvarname,red_perc = reduction_perc, wrapper = wrapper)
        xrfiltered = xrfiltered[heightvarname]
        
    volvalues = []

    for i in range(len(xrfiltered[name4d].values)):
        volvalues.append(np.nansum((xrfiltered.isel({name4d:i}).values))*pixelarea)

    df = pd.DataFrame({'date': xrfiltered[name4d].values, 
                       'value': volvalues,
                       'metric': 'volume'})

    return df

def growth_rate(df, datecolumn = None,valcolumn = 'value'):
    gr = [j-i for i, j in zip(df[valcolumn].values[:-1], 
                                df[valcolumn].values[1:])]
    nameslist = ['gr_t{}t{}'.format(
        (i+1),(i)) for i in range(0, len(gr)) ]

    if datecolumn is not None:
        namesdays = [(df[datecolumn].iloc[i+1] -df[datecolumn].iloc[i]) for i in range(0, len(gr)) ]
        grdf = pd.DataFrame({
        datecolumn:namesdays,
        'value':gr,
        'name':nameslist})
        grdf[datecolumn] = grdf.date.dt.days

    else:
        namesdays = nameslist
        grdf = pd.DataFrame({
        'value':gr,
        'name':nameslist})
    
    return grdf

class Phenomics:

    def check_dfphen_availability(self, phen = 'plant_height', **kwargs):
        """
        a functions to check if the pehnotype was already calculated, otherwise it
        will calculate the metric using default parameters
        ...
        Parameters
        ----------
        phen : str
        ...
        Returns
        -------
        pandas dataframe:
        """
        
        if phen in list(self._phenomic_summary.keys()):
            dfs = self._phenomic_summary[phen]
        else:
            if phen == 'plant_height':
                self.plant_height_summary(**kwargs)
                
            if phen == 'leaf_angle':
                self.leaf_angle_summary(**kwargs)
                
            if phen == 'volume':
                self.volume_summary(**kwargs)
            
            if phen in SPECTRAL_METRICS:
                self.splectral_reflectance(spindexes = phen)

            if phen == 'leaf_area':
                self.leaf_area(**kwargs)

            if phen == 'rosette_area':
                self.rosette_area(**kwargs)

            if phen == 'convex_hull':
                self.convex_hull_area()

            dfs = self._phenomic_summary[phen]

        return dfs


    def phenotype_growth_rate(self, phen = 'plant_height', valuecolname = 'value', **kwargs):

        name4d = list(self.xrdata.dims.keys())[0]
        dfs = self.check_dfphen_availability(phen, **kwargs)
        
        
        dfg = dfs.groupby('metric').apply(
            lambda x: growth_rate(x, datecolumn=name4d)).reset_index()
        dfg['metric'] = dfg['metric'] + dfg['name']
        
        self._phenomic_summary[phen+'_gr'] = dfg[
            ['date','value','metric']]
        
        return self._phenomic_summary[phen+'_gr']


    def earlystages_areas(self, refband = 'red', scalefactor = 100):

        dfla = self.leaf_area(refband = refband,scalefactor=scalefactor)
        dfra =self.rosette_area(refband = refband,scalefactor=scalefactor)
        dfch = self.convex_hull_area(refband = refband,scalefactor=scalefactor)

        return pd.concat([dfla,
                dfra,
                dfch], axis=0)

    def leaf_area(self, refband = 'red', scalefactor = 100):
        plantareaperdate = []
        xrdata = self.xrdata.copy()
        name4d = list(xrdata.dims.keys())[0]
        pixelsize = xrdata.attrs['transform'][0]*scalefactor
        
        for doi in range(len(xrdata.date.values)):
            initimageg = xrdata.isel(date =doi).copy()

            plantareaperdate.append(np.nansum(
                np.logical_not(np.isnan(initimageg[refband].values)))*pixelsize*pixelsize)

        self._phenomic_summary['leaf_area'] = pd.DataFrame({name4d:xrdata[name4d].values,
                      'value':plantareaperdate,
                      'metric': 'leaf_area'})
        
        return self._phenomic_summary['leaf_area']

    def convex_hull_area(self, refband = 'red', scalefactor = 100):
        convexhullimgs = []
        xrdata = self.xrdata[refband].copy().values
        name4d = list(self.xrdata.dims.keys())[0]
        pixelsize = self.xrdata.attrs['transform'][0]*scalefactor
        for doi in range(len(self.xrdata.date.values)):
            initimageg = xrdata[doi]
            initimageg[initimageg ==0 ] = np.nan
            initimageg[np.logical_not(np.isnan(initimageg))] = 1
            initimageg[np.isnan(initimageg)] = 0

            chull = convex_hull_image(initimageg, 
                                      offset_coordinates=False)
            convexhullimgs.append(np.nansum(chull)*pixelsize*pixelsize)

        
        self._phenomic_summary['convex_hull'] = pd.DataFrame(
            {name4d:self.xrdata[name4d].values,
                      'value':convexhullimgs,
                      'metric': 'convex_hull'})
        
        return self._phenomic_summary['convex_hull']

    def convex_hull_plot(self, refband = 'red', 
                         scalefactor = 100, figsize = (20,12),
                          saveplot = False,
                          outputpath = None, alphahull = 0.2,
                          size = 8,
                          ccenter = 'red',
                          addcenter = True):

        name4d = list(self.xrdata.dims.keys())[0]

        fig, ax = plt.subplots(figsize=figsize, ncols=len(self.xrdata[name4d].values),
                               nrows=1)
        
       
        pixelsize = self.xrdata.attrs['transform'][0]*scalefactor
        

        for doi in range(len(self.xrdata.date.values)):
            
            
            threebanddata = []
            for i in ['red', 'green','blue']:
                threebanddata.append(self.xrdata.isel(date=doi).copy()[i].data)
                

            threebanddata = np.dstack(tuple(threebanddata))/255
            imgpd = int(threebanddata.shape[0]*1.15)
            threebanddata = img_padding(threebanddata, imgpd)
            initimageg = threebanddata[:,:,0]
            initimageg[initimageg ==0 ] = np.nan
            initimageg[np.logical_not(np.isnan(initimageg))] = 1
            initimageg[np.isnan(initimageg)] = 0
            

            chull = convex_hull_image(initimageg, 
                                      offset_coordinates=False)
            
            c = getcenter_from_hull(initimageg)

            ax[doi].imshow(threebanddata)
            if addcenter:
                ax[doi].scatter( c[1], c[0], c = [ccenter], s = [size])

            ax[doi].imshow(chull, alpha = alphahull)
            area = np.nansum(chull)*pixelsize*pixelsize
            ax[doi].invert_xaxis()
            ax[doi].set_axis_off()
            ax[doi].set_title("CH area\n{} (cm2)".format(np.round(area,2)), size=28, color = 'r')
        
        if saveplot:
            if outputpath is None:
                outputpath = 'tmp.png'
            
            fig.savefig(outputpath)
            plt.close()



    def rosette_area(self, refband='red',scalefactor = 100 ,**kargs):
        
        xrdata = self.xrdata.copy()
        name4d = list(xrdata.dims.keys())[0]
        pixelsize = xrdata.attrs['transform'][0]*scalefactor

        distdates = []
        for doi in range(len(self.xrdata.date.values)):
            initimageg = xrdata.isel(date =doi).copy()
            leaflongestdist, ( xp,yp) = centerto_edgedistances_fromxarray(
                initimageg, wrapper='circle')
            distdates.append(
                leaflongestdist)

        self._phenomic_summary['rosette_area'] = pd.DataFrame(
            {name4d:self.xrdata[name4d].values,
                      'value':[i*i*pixelsize*pixelsize* math.pi for i in distdates],
                      'metric': 'rosette_area'})
        
        return self._phenomic_summary['rosette_area']


    def rosette_area_plot(self, 
                          refband = 'red', 
                          scalefactor = 100, 
                          figsize = (20,12),
                          saveplot = False,
                          outputpath = None):

        name4d = list(self.xrdata.dims.keys())[0]
        fig, ax = plt.subplots(figsize=figsize, 
                               ncols=len(self.xrdata[name4d].values),
                               nrows=1)
        
        xrdata = self.xrdata.copy()

        pixelsize = xrdata.attrs['transform'][0]*scalefactor
        
        xp = int(xrdata[refband].values.shape[2]/2)
        yp = int(xrdata[refband].values.shape[1]/2)

        
        for doi in range(len(self.xrdata.date.values)):
            initimageg = xrdata.isel(date =doi).copy()

            #if np.isnan(initimageg[refband].values[yp,xp]):
            #    yp, xp = getcenter_from_hull(initimageg[refband].copy().values)

            leaflongestdist, ( xp,yp) = centerto_edgedistances_fromxarray(
                initimageg, wrapper='circle')

            threebanddata = []
            for i in ['red', 'green','blue']:
                threebanddata.append(initimageg[i].data)
            imgpd = int(leaflongestdist*1.25)

            threebanddata = np.dstack(tuple(threebanddata))/255

            dif_height, dif_width = calculate_differencesinshape(threebanddata.shape[0],threebanddata.shape[1], 
                                                                imgpd)

            xp, yp = (xp+dif_width, yp+dif_height)
            threebanddata = img_padding(threebanddata, imgpd)
            draw_circle = plt.Circle((xp, yp), leaflongestdist,fill=False, color = 'r')
            
            ax[doi].add_artist(draw_circle)
            ax[doi].imshow(threebanddata)
            ax[doi].scatter(xp, yp, color = 'r' )

            ax[doi].invert_xaxis()
            ax[doi].set_axis_off()
            ax[doi].set_title("Rosette radious\n{} (cm)".format(
                np.round(leaflongestdist*pixelsize,2)), size=28, color = 'r')

        if saveplot:
            if outputpath is None:
                outputpath = 'tmp.png'
            
            fig.savefig(outputpath)
            plt.close()

    def plot_spindexes(self, spindexes = SPECTRAL_METRICS, **kargs):
        df = self.splectral_reflectance(spindexes)
        return tsquantiles_plot(df, **kargs)    

    def splectral_reflectance(self, spindexes = SPECTRAL_METRICS, quantiles = [0.25,0.5,0.75], shadowmask = None):
        tmpxarray = self.xrdata.copy()
        if type(shadowmask) == np.ndarray:
            tmpxarray = tmpxarray.copy().where(shadowmask,np.nan)
            self._xrshadowfiltered = tmpxarray

        spectraldf = []
        if type(spindexes) == list:
            for spindex in spindexes:
                df = get_df_quantiles(tmpxarray, varname= spindex, quantiles = quantiles)
                self._phenomic_summary[spindex] = df
                spectraldf.append(df)

        else:
            df = get_df_quantiles(tmpxarray, varname= spindexes, quantiles=quantiles)
            self._phenomic_summary[spindexes] = df
            spectraldf.append(df)

        return pd.concat(spectraldf, axis=0)


    def plant_height_summary(self, varname = 'z', quantiles = [0.25,0.5,0.75], 
                             reduction_perc = None, **kwargs):
        
        """
        a function to summarise the 2D heigth image into quantiles
        ...
        Parameters
        ----------
        varname : str, optional
            this indicates the height variable name, default 'z'
        quantiles: list, optional
            the quantiles in which the data must be summarized, the values must be 
            inside of a list
        reduction_perc: float, optional
            a value between 0 and 100, which determine the ratio that the image must be 
            reduced. Starting from the edges.
        ...
        Returns
        -------
        pandas dataframe:

        """

        self._ph_varname = varname
        
        if reduction_perc is not None:
            xrdata = get_filteredimage(self.xrdata, 
                                       channel= varname,
                                       red_perc = reduction_perc, **kwargs)
        else:
            xrdata = self.xrdata.copy()

        self._phenomic_summary[
            'plant_height'] = get_df_quantiles(xrdata, 
            varname= varname, quantiles=quantiles)
        
        return self._phenomic_summary['plant_height'] 

    def leaf_angle_summary(self, varname = 'leaf_angle', quantiles = [0.25,0.5,0.75]):
        """
        a function to summarise and the leaf angle 2D image into quantiles
        ...
        Parameters
        ----------
        varname : str, optional
            this indicates the leaf angle variable name, default 'leaf_angle'
        quantiles: list, optional
            the quantiles in which the data must be summarized, the values must be 
            inside of a list
        reduction_perc: float, optional
            a value between 0 and 100, which determines the ratio that the image must be 
            reduced. Starting from the edges.
        ...
        Returns
        -------
        pandas dataframe:

        """
        self._langle_varname = varname
        self._phenomic_summary[
            'leaf_angle'] = get_df_quantiles(self.xrdata, 
            varname= varname, quantiles=quantiles)

        return self._phenomic_summary['leaf_angle'] 

    def volume_summary(self, method = 'leaf_angle', **kargs):
        self._volume = calculate_volume(self.xrdata, method = method,**kargs)
        
        self._phenomic_summary['volume'] = self._volume

        return self._phenomic_summary['volume'] 


    def phenotype_ts_plot(self, phen = 'plant_height', **kargs):
        if phen == 'plant_height':
            df = self.plant_height_summary()
            plotts = tsquantiles_plot(df, yname = phen, **kargs)
        if phen == 'leaf_angle':
            df = self.leaf_angle_summary()
            plotts = tsquantiles_plot(df, yname = phen, **kargs)
        if phen == 'volume':
            df = calculate_volume(self.xrdata,**kargs)
            plotts = ts_plot(df,yname = phen)

        return plotts


    def all_phenotype_metrics(self, 
                              morphologicalmetrics = 'all',
                              spectralmetrics = 'all',
                              earlystage_metrics = 'all',
                              quantiles = [0.25,0.5,0.75]):


        morphodf = []
        spectraldf = []
        if morphologicalmetrics == 'all':

            for varname in MORPHOLOGICAL_METRICS:
                if varname != 'volume':
                    df = get_df_quantiles(self.xrdata, varname= varname, 
                                          quantiles = quantiles)
                    df['metric'] = varname
                
                else:
                    df = calculate_volume(self.xrdata, method = 'leaf_angle')
                morphodf.append(df)
            morphodf = pd.concat(morphodf, axis=0)

        if spectralmetrics == 'all':
            
            for varname in SPECTRAL_METRICS:
                
                df = get_df_quantiles(self.xrdata, varname= varname,
                                      quantiles=quantiles)
                df['metric'] = varname
                spectraldf.append(df)

            spectraldf = pd.concat(spectraldf, axis=0)


    def __init__(self,
                 xrdata,
                 dates_oi = None,
                 earlystages_filter = False,
                 earlystages_dates = None,
                 filter_method = None,
                 rf_onlythesedates = None,
                 summaryquantiles = [0.25,0.5,0.75],
                 days_earlystage = 15,
                 datedimname = None):
                 

        if datedimname is None:
            name4d = list(xrdata.dims.keys())[0]
        else:
            name4d = datedimname
            
        self._phenomic_summary = {}   
        self.dim_names = list(xrdata.dims.keys())
        self.quantiles_vals = summaryquantiles

        datepos = [i for i in range(len(self.dim_names)) if self.dim_names[i] == name4d][0]
        
        if len(self.dim_names)<3:
            raise ValueError('this functions was conceived to work only with multitemporal xarray')


        if dates_oi is None:
            dates_oi = list(range(len(xrdata[self.dim_names[datepos]].values)))

        if earlystages_dates is None:
            earlystagedate = 0
            for i, date in enumerate(xrdata[self.dim_names[datepos]].values):
                if (date - xrdata[self.dim_names[datepos]].values[0])/ np.timedelta64(1, 'D') >days_earlystage:
                    break
                earlystagedate = i
            earlystages_dates = xrdata[self.dim_names[datepos]].values[:earlystagedate]
            earlystages_dates = list(range(len(earlystages_dates)))


        if earlystages_filter:
            dates_oi = earlystages_dates
        self.xrdata = xrdata.isel({name4d:dates_oi}).copy()

        # apply filter to remove small objects that don't belong to the main body
        if filter_method is not None:
            from .xr_functions import filter_3Dxarray_usingradial, filter_3Dxarray_contourarea
            #if filter_method == 'radial':
            #    self.xrdata = filter_3Dxarray_usingradial(self.xrdata, 
            #                                            onlythesedates = rf_onlythesedates,
            #                                            anglestep = 1,
            #                                            nathreshhold=4)
            if filter_method == 'contourarea':
                self.xrdata = filter_3Dxarray_contourarea(self.xrdata)
        
        self.varnames = list(xrdata.keys())
    
        

def from_quantiles_dict_to_df(quantiles_dict: Dict, 
                              idvalue: Optional[str] = None) -> pd.DataFrame:
    """
    Converts a dictionary with quantiles data to a pandas DataFrame, adjusts variable names, 
    and restructures the DataFrame to have quantiles as columns and an optional ID value.

    Parameters
    ----------
    quantiles_dict : Dict
        The dictionary containing quantiles data, with keys as variable names.
    idvalue : Optional[str], optional
        An identifier value to be added to all rows in the output DataFrame, by default None.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the quantiles data restructured, having an ID value (if provided),
        and quantiles as columns for each variable.
    """
    
    
    df =  pd.DataFrame(quantiles_dict)
    
    df_long = pd.melt(df, 
                      value_vars=list(quantiles_dict.keys()), ignore_index=False).reset_index()
    
    varnames = ['{}_{}'.format(var, ind).replace('.','') for var, ind in zip(
        df_long.variable.values,df_long['index'].values)]
    
    df_long['variable'] = varnames
    idvalue = '0' if idvalue is None else idvalue
    df_long['id'] = idvalue
    
    return df_long.pivot(index=['id'],columns = 
                         "variable",values ="value").reset_index()
        

def calculate_quantiles(nparray, quantiles = [0.25,0.5,0.75]):
    """
    Calculates specified quantile values for each 1D array along the first dimension of a numpy array.

    Parameters
    ----------
    nparray : np.ndarray
        The input numpy array from which quantiles are calculated. It expects an array order of HW. if a 3D array is given
        the array order must be CHW.
    quantiles : List[float], optional
        The list of quantiles to calculate, by default [0.25, 0.5, 0.75].

    Returns
    -------
    List[Dict[float, float]]
        A list of dictionaries, with each dictionary containing the quantiles for each 1D array within the numpy array.
        The keys are the quantile values requested, and the values are the calculated quantiles.

    Examples
    --------
    >>> import numpy as np
    >>> nparray = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    >>> calculate_quantiles(nparray)
    [{0.25: 15.0, 0.5: 20.0, 0.75: 25.0}, {0.25: 45.0, 0.5: 50.0, 0.75: 55.0}, {0.25: 75.0, 0.5: 80.0, 0.75: 85.0}]
    """
    
    listqvalues = []
    if len(nparray.shape)>2:
        for i in range(nparray.shape[0]):
            datq = {q : np.nanquantile(nparray[i].flatten(),q = q) for q in quantiles}
            listqvalues.append(datq)
    else:
        listqvalues = {q : np.nanquantile(nparray.flatten(),q = q) for q in quantiles}
            
    return listqvalues


class PhenomicsDataCube(DataCubeProcessing):
    """
    A class for summarizing data cube, specifically tailored for
    generating quantile summaries of 2D and 3D image channels.

    Attributes
    ----------
    xrdata : xarray.Dataset
        The xarray dataset containing the phenomic data.
    _channel_names : List[str]
        A list of channel names available in the dataset.
    _array_order : str
        The order of dimensions in the dataset, e.g., 'CHW' for channels, height, width.
    _ndims : int
        The number of dimensions in the dataset.
    _navalue : float
        The value used in the dataset to represent N/A or missing data.
    """
    def __init__(self, xrdata: xarray.Dataset = None, metrics: dict = None, 
                 array_order:str = 'CHW', navalue: float = 0) -> None:
        """
        Initializes the PhenomicsFromDataCube instance with the provided xarray dataset and configuration.

        Parameters
        ----------
        xrdata : xarray.Dataset
            The xarray dataset containing the phenomic data.
        metrics : Optional[Dict], optional
            A dictionary of metrics for summarization, by default None.
        array_order : str, optional
            The order of dimensions in the dataset, e.g., 'CHW' for channels, height, width, by default 'CHW'.
        navalue : float, optional
            The value used in the dataset to represent N/A or missing data, by default np.nan.
        """
        self.xrdata = xrdata
        self._array_order = array_order
        self._navalue = navalue
        self._update_params()
    
    def __call__(self, xrdata: xarray.Dataset, 
                 channels: Optional[List[str]] = None,
                 vegetation_indices: Optional[List[str]] = None,
                 color_spaces: Optional[List[str]] = None,
                 quantiles: Optional[List[float]] = None,
                 rgb_channels: Optional[List[str]] = ['red','green','blue']) -> Dict[str, Dict[float, float]]:
        """
        Makes the instance callable, allowing it to directly operate on an xarray.Dataset
        to summarize its data into quantiles for specified channels.

        Parameters
        ----------
        xrdata : xarray.Dataset
            The xarray dataset to be analyzed and summarized.
        channel_names : Optional[List[str]], optional
            A list of channel names to be summarized. If None, all channels will be summarized, by default None.
        quantiles : Optional[List[float]], optional
            A list of quantiles to calculate for each specified channel. If None, defaults to median (0.5), by default None.
        rgbchannels : List[str], optional
            List of channel names representing RGB. Defaults to ['red', 'green', 'blue'].
        Returns
        -------
        Dict[str, List[Dict[float, float]]]
            A dictionary where keys are channel names and values are lists of dictionaries, each containing calculated quantile values.
        """
        
        self.xrdata = xrdata
        channels = [] if channels is None else channels
        color_spaces = [] if color_spaces is None else color_spaces
        vegetation_indices = [] if vegetation_indices is None else vegetation_indices
        
        ## check vi list
        vegetation_indices = vegetation_indices if isinstance(vegetation_indices, list) else [vegetation_indices]
        vitocalculate = [channelname for channelname in vegetation_indices if channelname in self._available_vi]
        if len(vitocalculate)>0:
            self.calculate_vegetation_indices(vi_list=vitocalculate, update_data=True)
            self._update_params()
            
        ## check colors
        color_spaces = color_spaces if isinstance(color_spaces, list) else [color_spaces]
        coloravail = list(self._available_color_spaces.keys())
        colortocalculate = [channelname for channelname in color_spaces if channelname in coloravail]
        color_channels = []
        if len(colortocalculate)>0:
            for i in colortocalculate:
                self.calculate_color_space(color_space = i, rgbchannels =rgb_channels , update_data=True)
                color_channels += self._available_color_spaces[i]
                
            self._update_params()
            
        channel_names = channels + vitocalculate + color_channels
        return self.summarise_into_quantiles(channel_names=channel_names, quantiles=quantiles)


    @property
    def _depth_dimname(self):
        """Identifies the name of the depth dimension in the dataset, excluding common spatial dimensions."""
        dimsnames = self.xrdata.dims.keys()
        if len(dimsnames) == 2:
            depthname = None
        else:
            depthname = [i for i in dimsnames if i not in ['x','y','longitude','latitude']][0]
        
        return depthname
    
        
    def summarise_into_quantiles(self,channel_names: Optional[List[str]] = None, quantiles: List[float] = None):
        """
        Summarizes the data of specified channels into quantiles for 2D and 3D images within the dataset.

        Parameters
        ----------
        channel_names : Optional[List[str]], optional
            A list of channel names to summarize. If None, all channels are used, by default None.
        quantiles : Optional[List[float]], optional
            A list of quantiles to calculate for each channel. If None, defaults to the median (0.5), by default None.

        Returns
        -------
        Dict[str, List[Dict[float, float]]]
            A dictionary where keys are channel names and values are lists of dictionaries, each containing calculated quantile values.
        """
        if not channel_names:
            channel_names = self._channel_names
        
        channel_data = {}
        for channel_name in channel_names:
            channel_name = self._channel_names[0] if not channel_name else channel_name
            
            if self._ndims == 2:
                channel_data[channel_name] = self._2d_array_summary(
                    self.xrdata, channel_name=channel_name, quantiles= quantiles)
            if self._ndims == 3:
                channel_data[channel_name] = self._3d_array_summary(
                    self.xrdata, channel_name=channel_name, quantiles= quantiles)
        
        return channel_data
    
    def _update_params(self):
        self._channel_names = None if not self.xrdata else list(self.xrdata.keys())
        self._ndims = None if not self.xrdata else len(list(self.xrdata.sizes.keys()))
        
    
    def _3d_array_summary(self, channel_name: str = None, quantiles: List[float] = None):
        """
        Summarizes 3D images into quantiles for a specific channel.

        Parameters
        ----------
        channel_name : str
            The name of the channel to summarize.
        quantiles : Optional[List[float]], optional
            The quantiles to calculate, by default None which calculates the median.

        Returns
        -------
        List[Dict[float, float]]
            A list of dictionaries with quantile values for each depth slice of the 3D image.

        Raises
        ------
        ValueError
            If the dataset's array order is not 'DCHW', indicating depth, channel, height, width.
        """
        if self._array_order != "DCHW":
            raise ValueError('Currently implemented only for "DCHW" array order.')
        
        datasummary = []
        for i in self.xrdata.dims[self._depth_dimname]:
            xrdata2d = self.xrdata.isel({self._depth_dimname: 0})
            datasummary.append(self._2d_array_summary(xrdata2d, channel_name))
        
        return datasummary
        
    
    def _2d_array_summary(self, xrdata: xarray.Dataset, channel_name: None, quantiles: List[float] = None):
        """
        Summarizes 2D images into quantiles for a specific channel.

        Parameters
        ----------
        xrdata : xarray.Dataset
            The xarray dataset to summarize.
        channel_name : str
            The name of the channel to summarize.
        quantiles : Optional[List[float]], optional
            The quantiles to calculate, by default None which calculates the median.

        Returns
        -------
        List[Dict[float, float]]
            A list of dictionaries with calculated quantile values.

        Raises
        ------
        AssertionError
            If the specified channel name is not in the datacube's channel names.
        """
        if not quantiles:
            quantiles = [0.5]

        assert channel_name in self._channel_names, "Channel name must be in the datacube."
        
        channel_image = xrdata[channel_name].values
        if not self._navalue:
            channel_image[channel_image == self._navalue] = np.nan
            
        datasummary = calculate_quantiles(channel_image, quantiles=quantiles)
        
        return datasummary
        

    