import cv2 
from PyQt5.QtMultimedia import *
from PyQt5.QtMultimediaWidgets import *
import sys
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import os
import time
import numpy as np
face_cascade = cv2.CascadeClassifier('haar_face.xml')

joker = cv2.imread('trollface_PNG6.png')
original_joker_h,original_joker_w,joker_channels = joker.shape

joker_gray = cv2.cvtColor(joker, cv2.COLOR_BGR2GRAY)
ret, original_mask = cv2.threshold(joker_gray, 10, 255, cv2.THRESH_BINARY_INV)
original_mask_inv = cv2.bitwise_not(original_mask)

cap = cv2.VideoCapture(0)
ret, img = cap.read()
img_h, img_w = img.shape[:2]

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x,y,w,h) in faces:
        face_w = w
        face_h = h
        face_x1 = x
        face_x2 = face_x1 + face_w
        face_y1 = y
        face_y2 = face_y1 + face_h

        joker_width = int(1.5 * face_w)
        joker_height = int(joker_width * original_joker_h  )
        joker_x1 = face_x2 - int(face_w/2) - int(joker_width/2)
        joker_x2 = joker_x1 + joker_width
        joker_y1 = face_y1 - int(face_h*1)
        joker_y2 = joker_y1 + joker_height 

        #check to see if out of frame
        if joker_x1 < 0:
            joker_x1 = 0
        if joker_y1 < 0:
            joker_y1 = 0
        if joker_x2 > img_w:
            joker_x2 = img_w
        if joker_y2 > img_h:
            joker_y2 = img_h

        #Account for any out of frame changes
        joker_width = joker_x2 - joker_x1
        joker_height = joker_y2 - joker_y1

        #resize joker to fit on face
        joker = cv2.resize(joker, (joker_width,joker_height), interpolation = cv2.INTER_AREA)
        mask = cv2.resize(original_mask, (joker_width,joker_height), interpolation = cv2.INTER_AREA)
        mask_inv = cv2.resize(original_mask_inv, (joker_width,joker_height), interpolation = cv2.INTER_AREA)

        #take ROI for joker from background that is equal to size of joker image
        roi = img[joker_y1:joker_y2, joker_x1:joker_x2]

        #original image in background (bg) where joker is not
        roi_bg = cv2.bitwise_and(roi,roi,mask = mask)
        roi_fg = cv2.bitwise_and(joker,joker,mask=mask_inv)
        dst = cv2.add(roi_bg,roi_fg)

        #put back in original image
        img[joker_y1:joker_y2, joker_x1:joker_x2] = dst


        break
    #display image
    cv2.imshow('img',img) 

    #if user pressed 'q' break
    if cv2.waitKey(1) == ord('q'): # 
        break;

cap.release() #turn off camera 
cv2.destroyAllWindows() #close all windows


