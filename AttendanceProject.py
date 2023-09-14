import cv2,numpy as np,face_recognition,os
from datetime import  datetime

path = 'ImagesAttendance_1'
image = []
classNames = []
mylist = os.listdir(path)
for cl in mylist:
    cur_img = cv2.imread(f'{path}/{cl}')
    images = image.append(cur_img)
    classNames.append(os.path.splitext(cl)[0])
    # print(classNames)
# print(mylist)

def Attendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readline()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            myDataList.append(entry[0])
            if name not in nameList:
                now=datetime.now()
                dtstring=now.strtime('%H : %M : %S')
                f.writelines(f'\n{name},{dtstring}')

                #Attendance(name)

def findencodings(image):
    encodelist = []
    for img in image :
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

encodeListKnown = findencodings(image)
print('Encoding is completed')

cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    freesCurrFrame=face_recognition.face_locations(imgS)
    encodeCurrFrame=face_recognition.face_encodings(imgS,freesCurrFrame)

    for encodeFaces,FaceLoc in zip(encodeCurrFrame,freesCurrFrame):
        matches=face_recognition.compare_faces(encodeListKnown,encodeFaces)
        faceDis=face_recognition.face_distance(encodeListKnown,encodeFaces)
        print(faceDis)
        matchIndex=np.argmin(faceDis)

        if matches[matchIndex]:
            name=classNames[matchIndex].upper()
            x1,y1,x2,y2=FaceLoc
            x1,y1,x2,y2=x1*4,y1*4,x2*4,y2*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_PLAIN,1,(0,0,200),2)

            cv2.imshow('webcam_1_wr',img)
            cv2.waitKey(1)

