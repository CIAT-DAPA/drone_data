import cv2
import matplotlib.pyplot as plt
import numpy as np

def draw_frame(img, bbbox, dictlabels = None, default_color = (255,255,255)):
    imgc = img.copy()
    for i in range(len(bbbox)):
        x1,x2,y1,y2 = bbbox[i]

        widhtx = abs(x1 - x2)
        heighty = abs(y1 - y2)

        start_point = (x1, y1)
        end_point = (x2,y2)
        if dictlabels is not None:
            color = dictlabels[i]['color']
            label = dictlabels[i]['label']
        else:
            label = ''
            color = default_color

        thickness = 4
        xtxt = x1 if x1 < x2 else x2
        ytxt = y1 if y1 < y2 else y2
        imgc = cv2.rectangle(imgc, start_point, end_point, color, thickness)
        if label != '':

            imgc = cv2.rectangle(imgc, (xtxt,ytxt), (xtxt + int(widhtx*0.8), ytxt - int(heighty*.2)), color, -1)
            
            imgc = cv2.putText(img=imgc, text=label,org=(xtxt + int(abs(x1-x2)/15),
                                                            ytxt - int(abs(y1-y2)/20)), 
                                                            fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1*((heighty)/200), color=(255,255,255), thickness=2)
            
    return imgc


def plot_segmenimages(img, maskimg, boxes = None, figsize = (10, 8), 
                      bbtype = None, only_image = False, inverrgbtorder = True,**kwargs):
    
    datato = img.copy()
    heatmap = cv2.applyColorMap(np.array(maskimg).astype(np.uint8), 
                                cv2.COLORMAP_PLASMA)
    
    output = cv2.addWeighted(datato, 0.5, heatmap, 1 - 0.75, 0)
    if boxes is not None:
        output = draw_frame(output, boxes, bbtype = bbtype, **kwargs)
    
    if only_image:
        fig = output
    else:
        
        # plot the images in the batch, along with predicted and true labels
        fig, ax = plt.subplots(nrows = 1, ncols = 3,figsize=figsize)
        #ax = fig.add_subplot(1, fakeimg.shape[0], idx+1, xticks=[], yticks=[])
        #fig, ax = plt.subplots(ncols = 3, nrows = 1, figsize = (14,5))
        #.swapaxes(0,1).swapaxes(1,2).astype(np.uint8)
        if inverrgbtorder:
            order = [2,1,0]
        else:
            order = [0,1,2]
            
        ax[0].imshow(datato[:,:,order],vmin=0,vmax=1)
        ax[0].set_title('Real',fontsize = 18)
        ax[1].imshow(maskimg,vmin=0,vmax=1)
        ax[1].set_title('Segmentation',fontsize = 18)

        ax[2].set_title('Overlap',fontsize = 18)
        ax[2].imshow(output[:,:,order])
            
        
    return fig
