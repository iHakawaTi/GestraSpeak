import cv2
import tensorflow as tf
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
from tempfile import TemporaryFile
import time
#from gtts import gTTS
import os
signed_text = pyttsx3.init()

signed_text.setProperty('rate', 200)

def read_pyttsx3(text):
    signed_text.say(text)
    signed_text.runAndWait()


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1, detectionCon=0.8)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
words = []
wordslist = []
folder = "Data/NO"
counter = 0
labels = ["NO", "YES" ,"OKAY","THANK YOU"]
while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            #print(prediction, index)
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
        #cv2.rectangle(imgOutput, (x - offset, y - offset-50),
         #             (x+ offset+w, y - offset-50+50), (220, 230, 0),cv2.FILLED)
        cv2.putText(imgOutput, labels[index], (x-offset, y -26), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 0) , 5 , cv2.LINE_AA)
        cv2.putText(imgOutput, labels[index], (x-offset, y -26), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2 )
        cv2.rectangle(imgOutput, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (127, 127, 127), 4)
        key = cv2.waitKey(1)
        if key == ord("s"): #pressing s saves a word into "words"
            words.append(labels[index])
            text = labels[index]
            read_pyttsx3(text)

        #cv2.imshow("ImageCrop", imgCrop)
        #cv2.imshow("ImageWhite", imgWhite)
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)

    #TTS code here, words list is ready

    print(words)


#Better text,Choose a better colors, add border to text if possible








