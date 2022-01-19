from keras.models import load_model,save_model
import keras
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from math import hypot
import os
import h5py

#loading model and body classifier
body_classifier = cv2.CascadeClassifier(r'F:\python\OpenCV\haarcascade_fullbody.xml')
classifier = load_model(r"F:\python\OpenCV\spd.h5")


#categories = ['Nude','Not Nude ']
img_paths = r"F:\python\OpenCV\Datas\test\images.jpg"
n = len(img_paths)
test_img = image.load_img(img_paths,target_size=(200,200))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis=0)
test_img = test_img.reshape(-1, 299,299,3)
prediction = classifier.predict(test_img)[0][0]
if(prediction>0.5).all():
    print("This image is not safe to use and it depicts Nudity")
elif(prediction<0.5).all():
    print("Safe to use")
else:
    print("Invalid Image")


# #capturing with webcam
# cap = cv2.VideoCapture(0)

# while True:
#   _, frame = cap.read()
#   labels = []
#   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#   bodies = body_classifier.detectMultiScale(gray,1.2,3)

#   #drawing rectangle around bodies
#   for (x,y,w,h) in bodies:
#     cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
#     #defining region of interest
#     roi_gray = gray[y:y+h,x:x+w]
#     roi_gray = cv2.resize(roi_gray,(299,299),interpolation = cv2.INTER_AREA)

#     #standardizing the region of interest and converting them into array
#     if np.sum([roi_gray])!=0:
#       roi = roi_gray.astype('float')/255.0
#       roi = img_to_array(roi)
#       roi = np.expand_dims(roi, axis=0)

#       #predicting the images fed
#       prediction = classifier.predict(roi)[0]
#       label = nlabels[prediction.argmax()]
#       label_position = (x,y-10)
#       cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_COMPLEX, 1, 2)
#     else:
#       cv2.putText(frame, 'No bodies', label_position, cv2.FONT_HERSHEY_COMPLEX, 1, 2)


#   #display
#   cv2.imshow('Humans',frame)

#   #stop if escape key is pressed
#   k = cv2.waitKey(30) & 0xff
#   if k==27:
#     break
  
# cap.release()
# cv2.destroyAllWindows()
