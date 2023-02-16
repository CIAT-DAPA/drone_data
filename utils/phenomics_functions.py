
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import xarray
import math
from skimage.morphology import convex_hull_image

from .xr_functions import filter_3Dxarray_usingradial, filter_3Dxarray_contourarea
from .gis_functions import centerto_edgedistances_fromxarray,get_filteredimage
from .image_functions import getcenter_from_hull


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
                                                    name4d = 'date'):
    


    if heightvarname not in list(xrdata.keys()):
        raise ValueError('the height variable is not in the xarray')
    

    pixelarea =(xrdata.attrs['transform'][0] *100) * (xrdata.attrs['transform'][0] *100)

    if method == 'leaf_angle' and leaf_anglename in list(xrdata.keys()):
        xrfiltered = xrdata.where((xrdata[leaf_anglename])>leaf_anglethresh,np.nan)[heightvarname].copy()
    elif (method == 'window'):
        xrfiltered = get_filteredimage(xrdata, heightvarname = heightvarname,red_perc = reduction_perc)
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

    def check_dfphen_availability(self, phen = 'plant_height'):
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
                self.plant_height_summary()
                
            if phen == 'leaf_angle':
                self.leaf_angle_summary()
                
            if phen == 'volume':
                self.volume_summary()
            
            if phen in SPECTRAL_METRICS:
                self.splectral_reflectance(spindexes = phen)

            if phen == 'leaf_area':
                self.leaf_area()

            if phen == 'rosette_area':
                self.rosette_area()

            if phen == 'convex_hull':
                self.convex_hull_area()

            dfs = self._phenomic_summary[phen]

        return dfs


    def phenotype_growth_rate(self, phen = 'plant_height', valuecolname = 'value'):

        name4d = list(self.xrdata.dims.keys())[0]
        dfs = self.check_dfphen_availability(phen)
        
        
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
                          outputpath = None):

        name4d = list(self.xrdata.dims.keys())[0]

        fig, ax = plt.subplots(figsize=figsize, ncols=len(self.xrdata[name4d].values),
                               nrows=1)
        
        xrdata = self.xrdata[refband].copy().values
        
        pixelsize = self.xrdata.attrs['transform'][0]*scalefactor
        for doi in range(len(self.xrdata.date.values)):
            initimageg = xrdata[doi]
            initimageg[initimageg ==0 ] = np.nan
            initimageg[np.logical_not(np.isnan(initimageg))] = 1
            initimageg[np.isnan(initimageg)] = 0
            chull = convex_hull_image(initimageg, 
                                      offset_coordinates=False)
            threebanddata = []
            for i in ['red', 'green','blue']:
                threebanddata.append(self.xrdata.isel(date=doi).copy()[i].data)
            threebanddata = np.dstack(tuple(threebanddata))/255

            ax[doi].imshow(threebanddata)

            ax[doi].imshow(chull, alpha = 0.15)
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
            leaflongestdist,_ = centerto_edgedistances_fromxarray(initimageg, anglestep=2, 
                                                                        nathreshhold = 3, refband = refband)
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

            leaflongestdist,( xp,yp) = centerto_edgedistances_fromxarray(initimageg, anglestep=2,
                                    nathreshhold = 3, refband = refband)
                                  
            area = leaflongestdist*leaflongestdist*pixelsize*pixelsize* math.pi
            threebanddata = []
            for i in ['red', 'green','blue']:
                threebanddata.append(self.xrdata.isel(date=doi).copy()[i].data)
            threebanddata = np.dstack(tuple(threebanddata))/255

            draw_circle = plt.Circle((xp, yp), leaflongestdist,fill=False, color = 'r')

            ax[doi].add_artist(draw_circle)
            ax[doi].imshow(threebanddata)
            ax[doi].scatter(xp, yp, color = 'r' )

            ax[doi].invert_xaxis()
            ax[doi].set_axis_off()
            ax[doi].set_title("Rosette area\n{} (cm2)".format(np.round(area,2)), size=28, color = 'r')

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
                             reduction_perc = None):
        
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
            xrdata = get_filteredimage(self.xrdata, heightvarname = 'z',red_perc = reduction_perc)
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
                 days_earlystage = 15):
                 

        self._phenomic_summary = {}   
        self.dim_names = list(xrdata.dims.keys())
        self.quantiles_vals = summaryquantiles

        if len(self.dim_names)<3:
            raise ValueError('this functions was conceived to work only with multitemporal xarray')


        if dates_oi is None:
            dates_oi = list(range(len(xrdata[self.dim_names[0]].values)))

        if earlystages_dates is None:
            earlystagedate = 0
            for i, date in enumerate(xrdata[self.dim_names[0]].values):
                if (date - xrdata[self.dim_names[0]].values[0])/ np.timedelta64(1, 'D') >days_earlystage:
                    break
                earlystagedate = i
            earlystages_dates = xrdata[self.dim_names[0]].values[:earlystagedate]
            earlystages_dates = list(range(len(earlystages_dates)))

        name4d = list(xrdata.dims.keys())[0]
        if earlystages_filter:
            dates_oi = earlystages_dates
        self.xrdata = xrdata.isel({name4d:dates_oi}).copy()

        # apply filter to remove small objects that don't belong to the main body
        if filter_method is not None:
            if filter_method == 'radial':
                self.xrdata = filter_3Dxarray_usingradial(self.xrdata, 
                                                        onlythesedates = rf_onlythesedates,
                                                        anglestep = 1,
                                                        nathreshhold=4)
            if filter_method == 'contourarea':
                self.xrdata = filter_3Dxarray_contourarea(self.xrdata)
        
        self.varnames = list(xrdata.keys())
    
        
