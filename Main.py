import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from flask import Flask
app = Flask(__name__)

@app.route("/")
def main():
    #Aqui va su código

    img = cv.imread('RXimg.png')
    kernel = np.ones((5,5),np.uint8)
    opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    mask = np.zeros(opening.shape[:2],np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (92,84,227,229)
    cv.grabCut(opening,mask,rect,bgdModel,fgdModel,5,cv.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img2 = opening*mask2[:,:,np.newaxis]


    titles = ['Original Image','opening','final']
    images = [img,opening,img2]
    for i in range(3):
       plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
       plt.title(titles[i])
       plt.xticks([]),plt.yticks([])

    plt.show()
    return img2