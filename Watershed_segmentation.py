# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 10:12:14 2021

@author: PC Chabrawal
"""

#watershed Segmentation 

import skimage
import cv2
from matplotlib import pyplot as plt
from skimage import filters
from skimage import io
import numpy as np
from skimage.transform import resize
import glob

img_number = 1

path =  "C:/Users/PC Chabrawal/Desktop/EWIS Project/EWIS Data/144105801-805/*.*"

for image in glob.glob(path):
    image = cv2.imread(image)
    image =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
   
    image_green = image[:,:,1]
    
    
    #plt.imshow(image_green,cmap="gray")


     



    #manually applying histogram

    #hist_img = image_green.flatten()
    #histogram = plt.hist(hist_img, bins=100,range=(0,255))
    #plt.show()

    #foreground = (image_green > 170)
    #background = (image_green < 170)

    #plt.imshow(foreground,cmap="gray")
    #plt.imshow(background,cmap="gray")
    #io.imshow(image_green)

    


    

    #apply Thresholdotsu for automatic detection of histogram 
    ret1,thres1 = cv2.threshold(image_green, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #plt.imshow(thres1,cmap="gray")



    
    #morpholoical operators to remove small white noise-(opening)
    #to remove white small holes we can apply closing



    #import numpy as np

    #kernel = np.ones((3,3),np.uint8)
    #opening = cv2.morphologyEx(thres1,cv2.MORPH_OPEN,kernel,iterations=1)

    #plt.imshow(opening,cmap="gray")

    #tried opening to reduce noise but there is not that much noise so no need



    

    #watershed segmentation



    #sure background area

    kernel=np.ones((3,3))
    sure_bg = cv2.dilate(thres1,kernel,iterations=1)
    #plt.imshow(sure_bg,cmap="gray")



    
    #tried dist_transform but result didnt appear as expected so erosion works better 
    #dist_transform = cv2.distanceTransform(thres1,cv2.DIST_L2,5)
    #ret, sure_fg = cv2.threshold(dist_transform,0.1*dist_transform.max(),255,0)
    #plt.imshow(sure_fg,cmap="gray")

    
    #sure foreground area


    sure_fg = cv2.erode(thres1,kernel,iterations=1)
    #plt.imshow(sure_fg,cmap="gray")

    #unknown area 

    unknown = np.subtract(sure_fg,sure_bg)
    #plt.imshow(unknown,cmap="gray")


    #cv2.imshow("background",sure_bg)
    #cv2.imshow("foreground",sure_fg)
    #cv2.imshow("unknown",unknown)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()



    #marker labelling

    ret2,markers = cv2.connectedComponents(sure_fg)
    #plt.imshow(markers)

    #markers assign value 0 to background but watershed read value 0 as unknown so we have to give different int value to background

    markers = markers + 1


    #mark the unknown with 0

    markers[unknown==255] = 0

    #plt.imshow(markers)

    markers = cv2.watershed(image,markers)
    #plt.imshow(markers)
    
    image=image[430:1870,190:2150]
    image_green=image_green[430:1870,190:2150]
    thres1=thres1[430:1870,190:2150]
    sure_bg=sure_bg[430:1870,190:2150]   
    sure_fg=sure_fg[430:1870,190:2150]
    markers=markers[430:1870,190:2150]

    figure = plt.figure(figsize=(8,8))

    ax1 = figure.add_subplot(2,3,1)
    ax1.imshow(image)
    ax1.set_title("Original Image")

    ax2 = figure.add_subplot(2,3,2)
    ax2.imshow(image_green)
    ax2.set_title("Green Channel Image")

    ax3 = figure.add_subplot(2,3,3)
    ax3.imshow(thres1,cmap="gray")
    ax3.set_title("Thresolded Image")
    
    ax4=figure.add_subplot(2,3,4)
    ax4.imshow(sure_bg,cmap="gray")
    ax4.set_title("Sure_Background")

    ax5 = figure.add_subplot(2,3,5)
    ax5.imshow(sure_fg,cmap="gray")
    ax5.set_title("Sure_Foreground")
    
    ax6=figure.add_subplot(2,3,6)
    ax6.imshow(markers)
    ax6.set_title("Watershed_Segmentation")
    
    
    plt.savefig("figure"+str(img_number)+".jpg",dpi=150)
    img_number +=1
    
    plt.show()
    
    
    
    
   
print("finish")



