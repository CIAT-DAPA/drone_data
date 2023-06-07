import cv2

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