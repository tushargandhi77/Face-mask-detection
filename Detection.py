# Implement Model in Live
import numpy as np
from keras.models import load_model
import cv2
from keras.preprocessing import image
import datetime

mymodel=load_model('mymodel.h5')

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while cap.isOpened():
    _,img = cap.read()
    face = face_cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=4)
    for(x,y,w,h) in face:
        face_img = img[y:y+h,x:x+w]
        cv2.imwrite('temp.jpg',face_img)
        test_image = image.load_img('temp.jpg',target_size=(150,150,3))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image,axis=0)
        pred = mymodel.predict(test_image)[0][0]
        if pred==1:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
            cv2.putText(img,'NO Mask',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),3)
        else:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
            cv2.putText(img,'Mask',((x+w)//2,y+h+20),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),3)
        datet = str(datetime.datetime.now())
        cv2.putText(img,datet,(400,400),cv2.FONT_HERSHEY_PLAIN,0.5,(255,255,255),1)

    cv2.imshow('img',img)
    if cv2.waitKey(1) == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
