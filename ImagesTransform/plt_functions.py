import matplotlib.pyplot as plt
import random
import numpy as np

import matplotlib.patches as patches
from numpy.core.fromnumeric import size

def plot_multiple_rgb_images(npimages, n_images=5):
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.labelsize'] = False
    plt.rcParams['ytick.labelsize'] = False
    plt.rcParams['xtick.top'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.right'] = False
    plt.rcParams['figure.figsize'] = [20, 5]
    idlist = list(range(len(npimages)))
    random.shuffle(idlist)
    # plot images
    for i in range(n_images):
        plt.subplot(1, n_images, i + 1)
        plt.imshow(npimages[idlist[i]])


def plot_single_image(npimages: np.dstack, idimage = 0, figsize = (12,10)):
     
    plt.figure(figsize=figsize)
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.labelsize'] = False
    plt.rcParams['ytick.labelsize'] = False
    plt.rcParams['xtick.top'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.right'] = False
    plt.imshow(npimages[idimage])

def plot_single_image_odlabel(npimages: np.dstack, bbcoords = None,figsize = (12,10), linewidth = 2, edgecolor = 'r')->None:
    fig, ax = plt.subplots(figsize=figsize)
    
    plt.rcParams['axes.grid'] = False
    plt.rcParams['xtick.labelsize'] = False
    plt.rcParams['ytick.labelsize'] = False
    plt.rcParams['xtick.top'] = False
    plt.rcParams['xtick.bottom'] = False
    plt.rcParams['ytick.left'] = False
    plt.rcParams['ytick.right'] = False
    ax.imshow(npimages)
    for i in range(len(bbcoords)):
        if bbcoords is not None and len(bbcoords[i])==4:
            
            
            x1,x2,y1,y2 = bbcoords[i]
            centerx = x1+np.abs((x1-x2)/2) 
            centery = y1+np.abs((y1-y2)/2) 
            
            rect = patches.Rectangle((x1, y1), abs(x2-x1), abs(y2-y1), linewidth=linewidth, edgecolor=edgecolor, facecolor='none')
            ax.scatter(x=centerx, y=centery, c='r', linewidth=2)
            ax.add_patch(rect)
    

    