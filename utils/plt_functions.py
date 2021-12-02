import numpy as np
import matplotlib
from utils import data_processing
import matplotlib.pyplot as plt
import plotly.graph_objs as go

def plot_categoricalraster(data, colormap='gist_rainbow', nodata=np.nan, fig_width=12, fig_height=8):

    data = data.copy()

    if not np.isnan(nodata):
        data[data == nodata] = np.nan

    catcolors = np.unique(data)
    catcolors = len([i for i in catcolors if not np.isnan(i)])
    cmap = matplotlib.cm.get_cmap(colormap, catcolors)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    im = ax.imshow(data, cmap=cmap)
    fig.colorbar(im)
    ax.set_axis_off()
    plt.show()

def plot_multibands_fromxarray(xarradata, bands, fig_sizex=12, fig_sizey=8):

    threebanddata = []
    for i in bands:
        banddata = xarradata[i].data
        if banddata.dtype == np.uint8 or banddata.dtype == np.uint16:
           banddata = banddata.astype(np.float16)

        banddata[banddata == xarradata.attrs['nodata']] = np.nan
        threebanddata.append(data_processing.scaleminmax(banddata))

    threebanddata = np.dstack(tuple(threebanddata))

    fig, ax = plt.subplots(figsize=(fig_sizex, fig_sizey))

    ax.imshow(threebanddata)

    ax.set_axis_off()
    plt.show()



def plot_3d_cloudpoints(xrdata, scale_xy = 1, nonvalue = 0):

    plotdf = xrdata.to_dataframe().copy()
    ycoords = np.array([float("{}.{}".format(str(i[0]).split('.')[0][-3:], str(i[0]).split('.')[1])) for i in plotdf.index.values])*scale_xy
    xcoords = np.array([float("{}.{}".format(str(i[1]).split('.')[0][-3:], str(i[1]).split('.')[1]))  for i in plotdf.index.values])*scale_xy
    zcoords = plotdf.z.values

    nonvaluemask = zcoords.ravel()>nonvalue

    ## plotly3d
    xyzrgbplot = go.Scatter3d(
        x=xcoords[nonvaluemask], 
        y=ycoords[nonvaluemask], 
        z=zcoords.ravel()[nonvaluemask],
        mode='markers',
        marker=dict(color=['rgb({},{},{})'.format(r,g,b) for r,g,b in zip(plotdf.red.values[nonvaluemask], 
                                                                          plotdf.green.values[nonvaluemask], 
                                                                          plotdf.blue.values[nonvaluemask])]))

    layout = go.Layout(margin=dict(l=0,
                               r=0,
                               b=0,
                               t=0),
                    scene=dict(
                     aspectmode='data'))

    data = [xyzrgbplot]
    fig = go.Figure(data=data, layout=layout)
    fig.show()


def plot_2d_cloudpoints(clpoints, figsize = (10,6), xaxis = "latitude"):
    

    indcolors = [[r/255.,g/255.,b/255.] for r,g,b in zip(
                    clpoints.iloc[:,3].values, 
                    clpoints.iloc[:,4].values,
                    clpoints.iloc[:,5].values)]

    plt.figure(figsize=figsize, dpi=80)

    if xaxis == "latitude":
        loccolumn = 1
    elif xaxis == "longitude":
        loccolumn = 0

    plt.scatter(clpoints.iloc[:,loccolumn],
                clpoints.iloc[:,2],
                c = indcolors)
    

    plt.show()