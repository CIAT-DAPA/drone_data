
import pickle
from .gis_functions import get_filteredimage
from .multipolygons_functions import single_vi_bsl_impt_preprocessing
from .data_processing import from_long_towide, data_standarization, minmax_scale
import itertools
from .phenomics_functions import Phenomics
from datetime import datetime


def get_scalervalues(allxrdata):
    msbands = ['blue_ms', 'green_ms', 'red_ms', 'edge', 'nir']
    rgbbands = ['blue','green','red']
    vilist=['rgbvi', 'ndvi', 'savi', 'ndre', 'gndvi', "rgbvi_rgb", "grvi_rgb"]
    standar_dict = {'ms':[0,0],
                    'rgb':[0,0],
                    'msminmax': [0,0],
                    'rgbminmax': [0,0]}
    
    datapervar = []
    for idpol in range(len(allxrdata)):
        datapervar.append(allxrdata[idpol][msbands].to_array().values.flatten())
        
    standar_dict['ms'][0] = np.nanmean(list(itertools.chain.from_iterable(datapervar)))
    standar_dict['ms'][1] = np.nanstd(list(itertools.chain.from_iterable(datapervar)))
    standar_dict['msminmax'][0] = np.nanmin(list(itertools.chain.from_iterable(datapervar)))
    standar_dict['msminmax'][1] = np.nanmax(list(itertools.chain.from_iterable(datapervar)))

    datapervar = []
    for idpol in range(len(allxrdata)):
        datapervar.append(allxrdata[idpol][rgbbands].to_array().values.flatten())
        
    standar_dict['rgb'][0] = np.nanmean(list(itertools.chain.from_iterable(datapervar)))
    standar_dict['rgb'][1] = np.nanstd(list(itertools.chain.from_iterable(datapervar)))
    standar_dict['rgbminmax'][0] = np.nanmin(list(itertools.chain.from_iterable(datapervar)))
    standar_dict['rgbminmax'][1] = np.nanmax(list(itertools.chain.from_iterable(datapervar)))

    datapervar = []
    #for idpol in range(len(allxrdata)):
    #    datapervar.append(allxrdata[idpol][vilist].to_array().values.flatten())
    for vi in vilist:
        for idpol in range(len(allxrdata)):
            datapervar.append(allxrdata[idpol][vi].to_numpy().flatten())
         
        standar_dict[vi] = [np.nanmean(list(itertools.chain.from_iterable(datapervar))),
                                 np.nanstd(list(itertools.chain.from_iterable(datapervar)))]
        
        
    featurestopredict = rgbbands + msbands + vilist

    standar_values = {}
    for feature in featurestopredict:
        if feature in rgbbands:
            standar_values[feature] = standar_dict['rgb']
        if feature in msbands:
            standar_values[feature] = standar_dict['ms']
        if feature in vilist:
            standar_values[feature] = standar_dict[feature]
            

    #bands_mean_std_scaler = get_meanstd_fromlistxarray(allxrdata)
    bands_mean_std_scaler = None
    return [bands_mean_std_scaler, standar_values]

def get_summaryperoxrdata(fnpath, surfacelayerdatepos = None, datestofilter = None, sufacered = None, imageryreduction = None, vilist=None, qsummary = [0.25,0.5,0.75],
                          ph_quantile = None, ph_reduction = None, 
                          datelabel = None, scalers = None, method = 'standarization', 
                          featurestoscale = None, features =None, idplantsuffix = '_rgb', source = 'ms'):
    
    
    with open(fnpath, "rb") as fn:
        xrdata = pickle.load(fn)
    
    
    redimagery = get_filteredimage(xrdata.copy(),clip_xarray= True, red_perc= imageryreduction, 
                                        wrapper= None)
        
    ## surface correction
    if surfacelayerdatepos is not None:
        soilsurface = get_filteredimage(xrdata.isel(date  =surfacelayerdatepos).copy(),clip_xarray= True, red_perc= sufacered)
        bsl = np.median(soilsurface.z.values)
        baseline = True       
        redimagery = single_vi_bsl_impt_preprocessing(redimagery,
                                                leaf_angle = False,
                                                bsl_value=bsl,
                                                baseline=baseline,
                                                vilist=vilist,
                                                overwritevi = True,
                        imputation = True,
                        bandstofill = ['red','green','blue'],
                        equalization = False)[0]
        
    ### get data frame
    if datelabel is not None:
        lastpointdata = redimagery.isel(date = datestofilter).expand_dims(
            dim = {'date':[datetime.strptime(datelabel,'%Y-%m-%d')]},axis=0)
    else:
        lastpointdata = redimagery.isel(date = datestofilter).expand_dims(
            dim = {'date':[redimagery.date.values[datestofilter]]},axis=0)
    if scalers is not None:
        if method == 'standarization':
            for feature in featurestoscale:
                lastpointdata[feature].values = data_standarization(lastpointdata[feature].values, 
                                                                  meanval=scalers[feature][0], stdval=scalers[feature][1])  
        if method == 'standarization_unique':
            for feature in featurestoscale:
                lastpointdata[feature].values = data_standarization(lastpointdata[feature].values, 
                                                                  meanval=scalers[source][0], stdval=scalers[source][1])  
        if method == 'minmax_unique':
            for feature in featurestoscale:
                lastpointdata[feature].values = minmax_scale(lastpointdata[feature].values,
                                                                  minval=scalers[source][0],
                                                                  maxval=scalers[source][1])  

        
    tpdata = get_summary_metrics(lastpointdata, 
                                    features = features, 
                                    quantiles =qsummary,
                                    ph_quantile = ph_quantile,
                                    ph_reduction = ph_reduction)
    tmpfn = fnpath[fnpath.index("id_plant_"):]
    #print(tmpfn)
    idplant = tmpfn[len("id_plant_"):tmpfn.index(idplantsuffix)]
    
    tpdata["id_plant"] = idplant

    tpdata = from_long_towide(tpdata.sort_values(by=['date']), 
                        indexname = "id_plant", 
                        values_columnname = 'value', 
                        metrics_colname='metric')
    #tpdata["place"] = "awaji2022"
    
    return tpdata
    
    
    
def get_summary_metrics(xrdata, features = None, 
                        quantiles =None,
                        ph_quantile = 0.7,
                        volume = True,
                        ph_reduction = 100):
    
    outputs= []
    gsphenomics = Phenomics(xrdata, 
                        filter_method= None, 
                        earlystages_filter=False,
                        datedimname = 'date'
                        )

    if features is None:
        features = list(xrdata.keys())
        
    dfph = gsphenomics.plant_height_summary(quantiles= [ph_quantile], 
                                            reduction_perc= ph_reduction)
    outputs = outputs + [dfph]
    #dfleafangle = gsphenomics.leaf_angle_summary(quantiles= [0.5])
    if volume:
        dfvolume = gsphenomics.volume_summary(method = 'window', 
                                            reduction_perc = ph_reduction)
        outputs = outputs + [dfvolume]
        
    if quantiles is not None:
                
        ## reflectance metrics
        dfreflectance = gsphenomics.splectral_reflectance(spindexes=features,
                                                        quantiles= quantiles)
        outputs = outputs + [dfreflectance]

    ## concat 
    output  = pd.concat(outputs)
    
    return output

