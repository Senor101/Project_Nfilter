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
#these 4 lines improves frame capture rates
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)                                               
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)                                                
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)

mymodel = load_model(r"D:\Projects\SPD\spd.h5")

while True:
    ret,frame = cap.read()
    resized = cv2.resize(frame,(800,600))
    cv2.imwrite('temp.jpg',resized)
    test_image=image.load_img('temp.jpg',target_size=(200,200,3))
    test_image=image.img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    pred=mymodel.predict(test_image)[0][0]
    #print(pred)
    if(pred>0.5).all():
        cv2.putText(resized, 'Nude', (50,50), cv2.FONT_HERSHEY_COMPLEX, 
                   1, (255,0,0), 2, cv2.LINE_AA)
    elif(pred<0.5).all():
        cv2.putText(resized, 'Not Nude', (50,50), cv2.FONT_HERSHEY_COMPLEX, 
                   1, (255,0,0), 2, cv2.LINE_AA)
    else:
        print("Invalid")
            
            
    cv2.imshow('humans',resized)
    #stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
      break
    
cap.release()
cv2.destroyAllWindows()
