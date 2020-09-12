# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 20:51:42 2020

@author: Divesh
"""
#https://towardsdatascience.com/https-medium-com-dilan-jay-face-detection-model-on-webcam-using-python-72b382699ee9
#https://github.com/amineHorseman/facial-expression-recognition-using-cnn/blob/master/predict-from-video.py
#https://github.com/jaydeepthik/facial-expression-recognition-webcam/blob/master/cv_cam_facial_expression.py
#importing libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np

#https://www.docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html#cascadeclassifier-load
#loading the haar cascade classifier and the model
face_classifier=cv2.CascadeClassifier(r'D:\PythonProjects\FaceDetect\haarcascade_frontalface_default.xml')
classifier=load_model(r'D:\PythonProjects\FaceDetect\EmotionDetector_teston6-2.h5')
class_labels=['Anger','Contempt','Disgust','Fear','Happy','Neutral','Sadness','Surprise']

#capturing the video feed from web cam
capture=cv2.VideoCapture(0)

while True:
    #reading the frames
    _, frame=capture.read()
    labels=[]
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #https://www.docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html#cascadeclassifier-detectmultiscale
    #loading the haar cascade classifer to detect faces 
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        #drawing a rectangle around the face and capturing the face area for prediction
        cv2.rectangle(frame,(x,y),(x+w,y+h),(51,255,153),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(128,128),interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray])!=0:
            #converting the captured face to a array of floats
            roi=roi_gray.astype('float')/255.0
            roi=img_to_array(roi)
            roi=np.expand_dims(roi,axis=0)
            #predicting from the model
            preds=classifier.predict(roi)[0]
            #classifying the prediction into a class
            label=class_labels[preds.argmax()] 
            label_position=(x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_DUPLEX,2,(51,255,153),2)
    cv2.imshow('Emotion Detector',frame)
    #quits the program when q is pressed
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
capture.release()
cv2.destroyAllWindows()
