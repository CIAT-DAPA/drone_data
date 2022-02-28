import numpy as np
import matplotlib
from utils import data_processing
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import math

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
    ax.invert_xaxis()
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


def plot_cluser_profiles(tsdata, ncluster, ncols = None, nrow = 2):
    
    n_clusters = np.unique(ncluster).max()+1
    sz = tsdata.shape[1]

    ncols = int(n_clusters/2)
    fig, axs = plt.subplots(nrow, ncols,figsize=(25,10))
    #fig, (listx) = plt.subplots(2, 2)

    maxy = tsdata.max() + 0.5*tsdata.std()
    it = 0
    for xi in range(nrow):
        for yi in range(ncols):
            for xx in tsdata[ncluster == it]:
                axs[xi,yi].plot(xx.ravel(), "k-", alpha=.2)

    
            axs[xi,yi].plot(tsdata[ncluster == it].mean(axis = 0), "r-")
            
            axs[xi,yi].set_title('Cluster {}, nplants {}'.format(it + 1, tsdata[ncluster == it].shape[0]))


            axs[xi,yi].set_ylim([0, maxy])

            it +=1


def plot_slices(data, num_rows, num_columns, width, height, rot= False, invertaxis = True):
    
    """Plot a montage of 20 CT slices"""
    #data list [nsamples, x, y]
    if rot:
        data = np.rot90(data)
    #data = np.transpose(data)
    data = np.reshape(data, (num_rows, num_columns, width, height))
    rows_data, columns_data = data.shape[0], data.shape[1]
    heights = [slc[0].shape[0] for slc in data]
    widths = [slc.shape[1] for slc in data[0]]
    fig_width = 12.0
    fig_height = fig_width * sum(heights) / sum(widths)
    f, axarr = plt.subplots(
        rows_data,
        columns_data,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": heights},
    )
    for i in range(rows_data):
        for j in range(columns_data):
            m = np.transpose(data[i][j])
            axarr[i, j].imshow(m, cmap="gray")
            axarr[i, j].axis("off")
            if invertaxis:
                
                axarr[i, j].invert_yaxis()
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


def plot_multitemporal_rgb(xarraydata, nrows = 2, ncols = None, 
                          figsize = (20,20), scale = 255., 
                          bands =['red','green','blue']):
    
    if ncols is None:
        ncols = math.ceil(len(xarraydata.date) / nrows)
    
    fig, axs = plt.subplots(nrows, ncols,figsize=figsize)
    cont = 0
    

    for xi in range(nrows):
        for yi in range(ncols):
            if cont < len(xarraydata.date):
                dataimg = xarraydata.isel(date=cont).copy()
                if scale == "minmax":
                    datatoplot = np.dstack([(dataimg[i].data - np.nanmin(dataimg[i].data)
                    )/(np.nanmax(dataimg[i].data) - np.nanmin(dataimg[i].data)) for i in bands])
                else:
                    datatoplot = np.dstack([dataimg[i].data for i in bands])/scale
                axs[xi,yi].imshow(datatoplot)
                axs[xi,yi].set_axis_off()
                axs[xi,yi].set_title(np.datetime_as_string(
                    xarraydata.date.values[cont], unit='D'))
                axs[xi,yi].invert_xaxis()
                cont+=1
            else:
                axs[xi,yi].axis('off')


def plot_multitemporal_cluster(xarraydata, nrows = 2, ncols = None, 
                          figsize = (20,20), 
                          band ='cluster',
                          ncluster = None, 
                          cmap = 'gist_ncar'):
                          
    if ncols is None:
        ncols = math.ceil(len(xarraydata.date) / nrows)
    
    fig, axs = plt.subplots(nrows, ncols,figsize=figsize)
    cont = 0
    if ncluster is None:
        ncluster = len(np.unique(xarraydata['cluster'].values))

    cmap = matplotlib.cm.get_cmap(cmap, ncluster)


    for xi in range(nrows):
        for yi in range(ncols):
            if cont < len(xarraydata.date):
                datatoplot = xarraydata.isel(date=cont)[band]

                im = axs[xi,yi].imshow(datatoplot, cmap = cmap)
                axs[xi,yi].set_axis_off()
                axs[xi,yi].set_title(np.datetime_as_string(xarraydata.date.values[cont], unit='D'))
                axs[xi,yi].invert_yaxis()
                cont+=1
            else:
                axs[xi,yi].axis('off')

    cbar_ax = fig.add_axes([.9, 0.1, 0.02, 0.7])
    fig.colorbar(datatoplot, cax=cbar_ax)
            