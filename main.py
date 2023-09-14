import cv2
import numpy as np
import face_recognition
import os
from AttendanceProject import mylist,classNames,encodeListKnown,faceDis,Attendance
os.sys.path

imgEl_name = face_recognition.load_image_file('ImagesBasiC/ElonMusk.jpeg')
imgEl_name = cv2.cvtColor(imgEl_name, cv2.COLOR_BGR2RGB)

imgTest=face_recognition.load_image_file('ImagesBasic/ElonMusk.jpeg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

#imgEl_name=cv2.cvtColor(imgEl_name,cv2.COLOR_BGR2RGB)


print(mylist, classNames,'Encodings Compeleted',Attendance())
print(faceDis)
cv2.imshow('webcam_1',img)
cv2.waitKey(1)

faceLoc = face_recognition.face_locations(imgEl_name)[0]
encodeElon = face_recognition.face_encodings(imgEl_name)[0]
#rect_1=cv2.boundingRect(imgEl_name)

img_rectangle=cv2.rectangle(imgTest ,(63,55),(138,129),(255,0,255),2)
img_rectangle_1=cv2.rectangle(imgEl_name,(63,55),(138,129),(255,0,255),2)
print (faceLoc)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
#cv2.rectangle((imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2))

results = face_recognition.compare_faces([encodeElon], encodeTest)
faceDis = face_recognition.face_distance([encodeElon], encodeTest)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)

cv2.imshow('ElonMusk', imgEl_name)
cv2.imshow('ElonMusk', imgTest)
cv2.waitKey(0)