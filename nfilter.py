import cv2
import numpy as np
from keras.models import load_model
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import image

#for mobile wireless camera capture
url = 'http://192.168.1.5:8080/video'
cap = cv2.VideoCapture(url)

#for webcam
#cap = cv2.VideoCapture(0)

mymodel = load_model('spd.h5')

while True:
    ret,frame = cap.read()
    frame = cv2.resize(frame,(200,200))
    cv2.imwrite('temp.jpg',frame)
    test_image=image.load_img('temp.jpg',target_size=(200,200,3))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    pred=mymodel.predict(test_image)[0][0]
    print(pred)
    if pred == 1:
      print("Nude")
    else:
      print("Not Nude")


    cv2.imshow('humans',frame)
    #stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
      break
    
cap.release()
cv2.destroyAllWindows()
