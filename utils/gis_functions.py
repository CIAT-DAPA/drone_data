import rasterstats as rs
import pandas as pd
import xarray


def get_data_perpoints(xrdata, gpdpoints, var_names=None, long=True):
    """

    :param xrdata:
    :param gpdpoints:
    :param var_names:
    :param long:
    :return:
    """

    if var_names is None:
        var_names = list(xrdata.keys())
    listtest = []
    for i in var_names:
        dataextract = rs.zonal_stats(gpdpoints,
                                     xrdata[i].data,
                                     affine=xrdata.attrs['transform'],
                                     # geojson_out=True,
                                     nodata=xrdata.attrs['nodata'],
                                     stats="mean")
        pdextract = pd.DataFrame(dataextract)
        pdextract.columns = [i]
        listtest.append(pdextract)
    pdwide = pd.concat(listtest, axis=1)
    pdwide['id'] = pdwide.index
    if long:
        output = pdwide.melt(id_vars='id')
    else:
        output = pdwide
    return output
