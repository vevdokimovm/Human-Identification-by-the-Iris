import numpy as np
import cv2
import os
import glob
 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
 
path = os.walk('./New_people')
 
j = 0
for address, _, path_ in path:
 
  for tmp in path_:
    imgg = cv2.imread(address+"/"+tmp)
    gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
      img = cv2.rectangle(imgg,(x,y),(x+w,y+h),(255,0,0),2)
      roi_gray = gray[y:y+h, x:x+w]
      roi_color = img[y:y+h, x:x+w]
      eyes = eye_cascade.detectMultiScale(roi_gray)
      i = 0;
      for (ex,ey,ew,eh) in eyes:
          cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
          roi = roi_color[ey:ey+eh, ex:ex+ew, :]
          if i == 0:
            len_ = len(glob.glob('/content/CASIA1/*'))
            os.mkdir("/content/CASIA1/{}".format(len_ + 1))
            for k in range(7):
              if k <= 2:
                cv2.imwrite('/content/CASIA1/{}/lol_1_{}.jpg'.format(len_ + 1, k), roi)
              else:
                cv2.imwrite('/content/CASIA1/{}/lol_2_{}.jpg'.format(len_ + 1, k), roi)
          i += 1
      j += 1
 
 
 
