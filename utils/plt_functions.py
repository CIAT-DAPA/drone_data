import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objs as go

import math

def scaleminmax(values):
    return ((values - np.nanmin(values)) /
            (np.nanmax(values) - np.nanmin(values)))


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

def plot_multibands_fromxarray(xarradata, bands, fig_sizex=12, fig_sizey=8, xinverse = True):
    """
    This function will plot an xarray data in RGB format

    Parameters:
    ----------
    """
    threebanddata = []
    for i in bands:
        banddata = xarradata[i].data
        if banddata.dtype == np.uint8 or banddata.dtype == np.uint16:
           banddata = np.asarray(banddata, dtype=np.float64)

        banddata[banddata == xarradata.attrs['nodata']] = np.nan
        threebanddata.append(scaleminmax(banddata))

    threebanddata = np.dstack(tuple(threebanddata))

    fig, ax = plt.subplots(figsize=(fig_sizex, fig_sizey))

    ax.imshow(threebanddata)
    if xinverse:
        ax.invert_xaxis()
        
    ax.set_axis_off()
    plt.show()



def plot_3d_cloudpoints(xrdata, scale_xy = 1, nonvalue = 0, zaxisname = 'z', 
                        rgb_bandnames = ['red','green', 'blue']):
    """
    this function creates a 3D interactive function from a 2d image using xarra data

    Parameters:
    ----------
    xrdata : xarray data
    scale_xy: int, optional
        factor to scale the x and y axis
    nonvalue: int, optional
        a value that represent na pixels
    zaxisname: str, optional
        the z variable name
        
    """

    plotdf = xrdata.to_dataframe().copy()
    ycoords = np.array([float("{}.{}".format(str(i[0]).split('.')[0][-3:], str(i[0]).split('.')[1])) for i in plotdf.index.values])*scale_xy
    xcoords = np.array([float("{}.{}".format(str(i[1]).split('.')[0][-3:], str(i[1]).split('.')[1]))  for i in plotdf.index.values])*scale_xy
    zcoords = plotdf[zaxisname].values

    nonvaluemask = zcoords.ravel()>nonvalue

    ## plotly3d
    xyzrgbplot = go.Scatter3d(
        x=xcoords[nonvaluemask], 
        y=ycoords[nonvaluemask], 
        z=zcoords.ravel()[nonvaluemask],
        mode='markers',
        marker=dict(color=['rgb({},{},{})'.format(r,g,b) for r,g,b in
                           zip(plotdf[rgb_bandnames[0]].values[nonvaluemask], 
                               plotdf[rgb_bandnames[0]].values[nonvaluemask], 
                               plotdf[rgb_bandnames[0]].values[nonvaluemask])]))

    layout = go.Layout(margin=dict(l=0,
                               r=0,
                               b=0,
                               t=0),
                    scene=dict(
                     aspectmode='data'))

    data = [xyzrgbplot]
    fig = go.Figure(data=data, layout=layout)
    fig.show()


def plot_2d_cloudpoints(clpoints, figsize = (10,6), xaxis = "latitude", fontsize = 18):
    """
    This function create a fron or perfil plot from a cloud points file

    Parameters:
    ----------
    clpoints: pandas dataframe
        dataframe that contains the points values
    """

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
    plt.yticks(fontsize=fontsize)
    plt.ylabel('Altitude above sea level (m)\n', fontsize=fontsize+3)

    plt.xticks(fontsize=fontsize)
    plt.xlabel(f'{xaxis}\n', fontsize=fontsize+3)

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


def minmax_xarray(xrdata):

    for i in xrdata.keys():
        xrdata[i].values = np.array(
            (xrdata[i].values - np.nanmin(xrdata[i].values))/(
                np.nanmax(xrdata[i].values) - np.nanmin(xrdata[i].values)))
    return xrdata


def plot_multibands(xrdata, num_rows = 1, num_columns = 1, 
                    chanels_names = None,
                    figsize = [10,10], 
                    cmap = 'viridis', 
                    minmaxscale = True,
                     **kwargs):
    
    """
    create a figure showing multiple xarray variables
    ----------
    xrdata : Xarray data
    num_rows : int, optional
        set number of rows
    num_columns : int, optional
        set number of rows
    chanels_names : list of string, optional
        a list that has the variables names to be plot.    
    figsize : tuple, optional
        A tuple (width, height) of the figure in inches. 
    cmap : str, optional
        a matplotlib colormap name.
    fontsize : int, optional
        a number for setting legend title size.
    legfontsize : int, optional
        a number for setting legend title size.
    legtickssize : int, optional
        a number for setting legend ticks size.
    colorbar: float, optional
        if the plot will include a colorbar legend
    minmaxscale = float, optional
        if the array will be scaled using a min max scaler
    Returns
    -------
    """    

    xrdatac = xrdata.copy()
    if chanels_names is not None:
        xrdatac = xrdatac[chanels_names]
        
    if minmaxscale:
        xrdatac = minmax_xarray(xrdatac).to_array().values
    else:
        xrdatac = xrdatac.to_array().values


    return plot_multichanels(xrdatac,num_rows = num_rows, 
                      num_columns = num_columns, 
                      figsize = figsize, 
                      chanels_names = list(xrdata.keys()),
                      cmap = cmap,
                       **kwargs)

import matplotlib.pyplot as plt

def plot_multichanels(data, num_rows = 2, 
                     num_columns = 2, figsize = [10,10],
                     label_name = None,
                     chanels_names = None, 
                     cmap = 'viridis', 
                     fontsize=12, 
                     legfontsize = 15,
                     legtickssize = 15,
                     colorbar = True, vmin = None, vmax = None,
                     newlegendticks = None,
                     fontname = "Arial",
                     invertaxis = True):
    """
    create a figure showing one channel or multiple channels
    ----------
    data : Numpy array
    num_rows : int, optional
        set number of rows
    num_columns : int, optional
        set number of rows
    figsize : tuple, optional
        A tuple (width, height) of the figure in inches. 
    label_name : str, optional
        an string value for the colorbar legend.
    chanels_names : list of string, optional
        a list with the labels for each plot.
    cmap : str, optional
        a matplotlib colormap name.
    fontsize : int, optional
        a number for setting legend title size.
    legfontsize : int, optional
        a number for setting legend title size.
    legtickssize : int, optional
        a number for setting legend ticks size.
    colorbar: float, optional
        if the plot will include a colorbar legend

    Returns
    -------
    """     
                
    import matplotlib as mpl
    if chanels_names is None:
        chanels_names = list(range(data.shape[0]))

    fig, ax = plt.subplots(nrows=num_rows, ncols=num_columns, figsize = figsize)
    
    count = 0
    vars = chanels_names
    cmaptxt = plt.get_cmap(cmap)
    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    
    fontmainfigure = {'family': fontname,
        'color':  'black',
        'weight': 'normal',
        'size': fontsize,
        }

    fontlegtick = {'family': fontname,
        'color':  'black',
        'weight': 'normal',
        'size': legtickssize,
        }
    
    fontlegtitle = {'family': fontname,
        'color':  'black',
        'weight': 'normal',
        'size': legfontsize,
        }
    
    for j in range(num_rows):
        for i in range(num_columns):
            if count < len(vars):

                if num_rows>1 and num_columns > 1:
                    ax[j,i].imshow(data[count], cmap=cmaptxt, vmin=vmin, vmax=vmax)
                    ax[j,i].set_title(vars[count], fontdict=fontmainfigure)
                    if invertaxis:
                        ax[j,i].invert_xaxis()
                    ax[j,i].set_axis_off()
                elif num_rows == 1 or num_columns == 1:
                    ax[i].imshow(data[count], cmap=cmaptxt, vmin=vmin, vmax=vmax)
                    ax[i].set_title(vars[count], fontdict=fontmainfigure)
                    ax[i].invert_xaxis()
                    ax[i].set_axis_off()

                count +=1
            else:
                if num_rows>1:
                    ax[j,i].axis('off')
                else:
                    ax[i].axis('off')
    #cbar = plt.colorbar(data.ravel())
    #cbar.set_label('X+Y')
    #cmap = mpl.cm.viridis
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    if colorbar:
        cbar_ax = fig.add_axes([0.91, 0.15, 0.03, 0.7])
        cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    ax=ax, #orientation='vertical',
                    cax=cbar_ax,
                    pad=0.15)
        cb.ax.tick_params(labelsize=legtickssize)
        if label_name is not None:
            cb.set_label(label=label_name, fontdict=fontlegtitle)
        if newlegendticks:
            cb.ax.get_yaxis().set_ticks([])
            for j, lab in enumerate(newlegendticks):
                cb.ax.text(vmax, (7.2 * j + 2) / (vmax+3), lab,
                           ha='left', va='center',fontdict=fontlegtick)

            
    return fig,ax




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


def plot_multitemporal_rgbarray(arraydata, nrows = 2, ncols = None, 
                          figsize = (20,20), scale = 255.,
                          datelabes = None,
                          #bands =['red','green','blue'],
                          depthpos = 0,
                          savedir = None,
                          fontsize = 15,
                          titlelabel = True,
                          invertaxis = False,
                          addcenter = False,
                          colorcenter = "red",
                          sizecenter = 8,
                          fontname = "Arial"):
    """
    create a figure showing one ataked multiband figure
    ----------
    Params:

    arraydata : arra data (D x C x H x W)
    nrows : int
        set number of rows
    ncols : int, optional
        set number of rows
    figsize : tuple, optional
        A tuple (width, height) of the figure in inches. 
    bands: list, optional
        A list that contains which three bands will represent the RGB channels
    fontsize : int, optional
        a number for setting legend title size.
    savedir: str, optional
        a directory path where will be used to save the image
    titlelabel: boolean, optional
        add title to each panel
    Returns
    -------
    fig: a matplotlib firgure
    """
    
    fontmainfigure = {'family': fontname,
        'color':  'black',
        'weight': 'normal',
        'size': fontsize,
        }


    if ncols is None:
        ncols = math.ceil(arraydata.shape[depthpos] / nrows)
    
    fig, axs = plt.subplots(nrows, ncols,figsize=figsize)
    cont = 0
    mtdata = arraydata.copy()
    
    if depthpos == 1:
        mtdata = mtdata.swapaxes(0,1)
        
    if datelabes is None:
        datelabes = ['']*arraydata.shape[0]
        
        
    for xi in range(nrows):
        for yi in range(ncols):
            if cont < mtdata.shape[0]:
                dataimg = mtdata[cont].copy()
                if scale == "minmax":
                    datatoplot = np.dstack([(dataimg[i].data - np.nanmin(dataimg[i].data)
                    )/(np.nanmax(dataimg[i].data) - np.nanmin(dataimg[i].data)) for i in range(dataimg.shape[0])])
                else:
                    datatoplot = np.dstack([dataimg[i].data for i in range(dataimg.shape[0])])/scale
                if nrows > 1:
                    axs[xi,yi].imshow(datatoplot)
                    axs[xi,yi].set_axis_off()
                    if titlelabel:
                        axs[xi,yi].set_title(datelabes[cont], fontdict=fontmainfigure)
                    if invertaxis:
                        axs[xi,yi].invert_xaxis()
                    
                    if addcenter:
                        
                        c = [datatoplot.shape[0]//2, datatoplot.shape[1]//2]
                        
                        axs[xi,yi].scatter( c[1], c[0], c = [colorcenter], s = [sizecenter])

                    cont+=1
                else:
                    axs[yi].imshow(datatoplot)
                    axs[yi].set_axis_off()
                    if titlelabel:
                        axs[yi].set_title(datelabes[cont], fontdict=fontmainfigure)
                    if invertaxis:
                        axs[yi].invert_xaxis()

                    if addcenter:
                        c = [datatoplot.shape[2]//2, datatoplot.shape[1]//2]
                        axs[yi].scatter( c[1], c[0], c = [colorcenter], s = [sizecenter])

                    cont = yi+1
                
            else:
                axs[xi,yi].axis('off')

    if savedir is not None:
        fig.savefig(savedir)
    
    return fig

def plot_multitemporal_rgb(xarraydata, nrows = 2, ncols = None, 
                          figsize = (20,20), scale = 255., 
                          bands =['red','green','blue'],
                          savedir = None,
                          fontsize = 15,
                          titlelabel = True,
                          depthpos = None,
                          name4d = 'date',
                          invertaxis = True,
                          **kwargs):
    """
    create a figure showing one ataked multiband figure
    ----------
    Params:

    xarraydata : Xarray data
    nrows : int
        set number of rows
    ncols : int, optional
        set number of rows
    figsize : tuple, optional
        A tuple (width, height) of the figure in inches. 
    bands: list, optional
        A list that contains which three bands will represent the RGB channels
    fontsize : int, optional
        a number for setting legend title size.
    savedir: str, optional
        a directory path where will be used to save the image
    titlelabel: boolean, optional
        add title to each panel
    Returns
    -------
    fig: a matplotlib firgure
    """    
    if ncols is None:
        ncols = math.ceil(len(xarraydata.date) / nrows)
    
    #fig, axs = plt.subplots(nrows, ncols,figsize=figsize)
    #cont = 0
    xarraydims = list(xarraydata.dims.keys())
    if depthpos is None:
        depthpos = [i for i in range(len(xarraydims)) if xarraydims[i] ==name4d][0]+ 1
        
    datelabesl = [np.datetime_as_string(i, unit='D') for i in xarraydata[name4d].values] 
    
    if bands is not None:
        xrdatac = xarraydata[bands].to_array().values.copy()
    else:
        xrdatac = xarraydata.to_array().values.copy()
    
    
    fig = plot_multitemporal_rgbarray(xrdatac, ncols = ncols, nrows=nrows,
                          figsize = figsize, scale = scale,
                          datelabes = datelabesl,
                          
                          depthpos = depthpos,
                          savedir = savedir,
                          fontsize = fontsize,
                          titlelabel = titlelabel,
                          invertaxis = invertaxis,
                          **kwargs)
    
    
    return fig

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



def adding_phfigure(altref, indcolors, xaxisref, yphreference, var, fontsize, vmin, vmax, vmaxl, 
                    ax = None, yfontize = 18,
                    hmin = None, hmax = None,
                    fontname = 'Helvetica'):
    
    xvalues = altref.iloc[:,0]
    xvalues = xvalues - hmin
    ax.scatter(xvalues*100,
                                altref.iloc[:,2],
                                c = indcolors)
    
    xaxisref = np.nanquantile(xvalues*100,0.05)
               
    #ax[j,i].axhline(y = yphreference, color = 'black', linestyle = '-')
    ax.plot((xaxisref, xaxisref), (yphreference,yphreference), color = 'black', linestyle = '-')
    ax.plot((xaxisref, xaxisref), (0,yphreference), color = 'red', linestyle = '-', linewidth=10)
    ax.axhline(y = 0, color = 'black', linestyle = '-')
    ax.set_title(var, fontsize=fontsize, fontweight='bold', fontname = fontname)
    #ax.set_xticks([])
    ax.yaxis.set_ticks(np.arange(0, (vmax-(vmax%10))+20, 10))
    ax.grid(color = 'gray', linestyle = '--', linewidth = 1,axis = 'y')
    ax.yaxis.set_tick_params(labelsize=yfontize)
    ax.xaxis.set_tick_params(labelsize=yfontize-1)
    ax.set_ylim(vmin, vmax+vmaxl)
    ax.set_xlim((hmin-hmin)*100,(hmax-hmin)*100)

    return ax


def plot_heights(xrdata, num_rows = 2, 
                     num_columns = 2, 
                     figsize = [10,10],
                     height_name = 'z',
                     bsl = None,
                     chanels_names = None, 
                     label_name = 'Height (cm)', 
                     fontsize=18,
                     scalez = 100,
                     phquantile = 0.5, 
                     fig = None,
                     ax = None,
                     vmin = None,
                     vmax = None,
                     reduction_perc = None,
                     yfontize = 18,
                    fontname = 'Helvetica'):
    
    """create a figure showing a 2d profile from a 3d image reconstruction
    ----------
    Params:
    
    xrdata : xarray data
    num_rows : int, optional
        set number of rows
    num_columns : int, optional
        set number of rows
    figsize : tuple, optional
        A tuple (width, height) of the figure in inches. 
    height_name : str
        column name assigned for z axis.
    bsl : float, optional
        dfault value for soil reference, if this is not given, it will be calculated from the first image.
    chanels_names : str list, optional
        a list with the labels for each plot..
    label_name : str, optional
        y axis label.
    fontsize: int, optional
        font size
    scalez: int, optional
        integer number for scaling height values
    phquantile: int optional
        decimal [0-1] determines the quantile for the height reference

    Returns
    matplotlib figure
    -------
    """  
    if chanels_names is None:
        chanels_names = [np.datetime_as_string(i, unit='D') for i in xrdata.date.values ]

    if fig is None and ax is None:
        fig, ax = plt.subplots(nrows=num_rows, ncols=num_columns, figsize = figsize)
    #else:
    #    ax = fig.subplots(nrows=num_rows, ncols=num_columns, sharey=True)
    count = 0
    vars = chanels_names
    xrdatac = xrdata.copy()
    #xrdatac = xrdatac[height_name].where(xrdatac[height_name] > 0, np.nan)
    if bsl is not None:
        xrdatac[height_name] = (xrdatac[height_name] - bsl)
        xrdatac[height_name] = xrdatac[height_name].where(xrdatac[height_name] > 0, np.nan)
        xrdatac[height_name] = xrdatac[height_name]*scalez 
    if reduction_perc is not None:
        xrdatac2 = get_filteredimage(xrdatac, heightvarname = 'z',red_perc = reduction_perc, clip_xarray=True)
        data = xrdatac[height_name].values    
    else:
        data = xrdatac[height_name].values

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    vmaxl = 1*(np.nanstd(data))
    hmin, hmax = np.min(xrdatac.x.values),np.max(xrdatac.x.values)
    #print(np.unique(altref.iloc[:,1]/100))
    for j in range(num_rows):
        for i in range(num_columns):
            if count < len(vars):
                xrtestdf = xrdatac.isel(date = count).to_dataframe()
                altref = xrtestdf.reset_index().loc[:,('x','y','z','red','green','blue')].dropna() 
                indcolors = [[r/255.,g/255.,b/255.] for r,g,b in zip(
                altref.iloc[:,3].values, 
                altref.iloc[:,4].values,
                altref.iloc[:,5].values)]
                xaxisref = np.nanquantile(altref.iloc[:,0],0.4)
                
                
                yphreference = np.nanquantile(altref[height_name].values[altref[height_name].values>0],
                                              phquantile )
                
                
                if num_rows>1:
                    ax[j,i] = adding_phfigure(altref, indcolors, xaxisref, 
                                              yphreference, vars[count], 
                                              fontsize, vmin, vmax, vmaxl, ax = ax[j,i],
                                              hmin = hmin, hmax = hmax, fontname =fontname,
                                              yfontize= yfontize)
                    
                else:
                    ax[i] = adding_phfigure(altref, indcolors, xaxisref, 
                                            yphreference, vars[count], 
                                            fontsize, vmin, vmax, vmaxl, ax = ax[i],
                                            hmin = hmin, hmax = hmax, fontname =fontname,
                                            yfontize= yfontize)

                count +=1
            else:
                if num_rows>1:
                    ax[j,i].axis('off')
                else:
                    ax[i].axis('off')
    # Adding a plot in the figure which will encapsulate all the subplots with axis showing only
    fig.add_subplot(1, 1, 1, frame_on=False)

    # Hiding the axis ticks and tick labels of the bigger plot
    plt.tick_params(labelcolor="none", bottom=False, left=False)

    # Adding the x-axis and y-axis labels for the bigger plot
    #plt.xlabel('Common X-Axis', fontsize=15, fontweight='bold')
    plt.ylabel(label_name + '\n', fontsize=int(fontsize*1.5), fontweight='bold')
    plt.xlabel('Longitude (cm)', fontsize=int(fontsize*1.5), fontweight='bold')


    #plt.show()
    #cbar = plt.colorbar(data.ravel())
    #cbar.set_label('X+Y')
    #cmap = mpl.cm.viridis
    return fig, ax