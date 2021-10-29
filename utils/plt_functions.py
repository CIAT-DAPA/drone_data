import numpy as np
import matplotlib
from utils import data_processing
import matplotlib.pyplot as plt


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
        banddata[banddata == xarradata.attrs['nodata']] = np.nan
        threebanddata.append(data_processing.scaleminmax(banddata))

    threebanddata = np.dstack(tuple(threebanddata))

    fig, ax = plt.subplots(figsize=(fig_sizex, fig_sizey))

    ax.imshow(threebanddata)

    ax.set_axis_off()
    plt.show()

