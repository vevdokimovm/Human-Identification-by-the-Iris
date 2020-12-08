import numpy as np
import cv2
import os
 
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
 
path = os.walk('./Faces') # path to your current folder, you can change it 
 
 # adress - path to current folder
 # _ - list of paths to all directories in the current folder (adress)
 # path_ - path to every file in current folder
 
j = 1
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
          if i == 0 or i == 1:
            cv2.imwrite('Eyes/eye_{}.jpg'.format(j), roi)
          else :
            break
          i += 1
          j += 1

