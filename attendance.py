import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'trained'
images = []
classNames = []
mylist = os.listdir(path)

for cls in mylist:
    current_img = cv2.imread(f'{path}/{cls}')
    images.append(current_img)
    classNames.append(os.path.splitext(cls)[0])

# print(classNames)

def findEncodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode_img = face_recognition.face_encodings(img)[0]
        encode_list.append(encode_img)
    return encode_list

def markAttendance(name):
    with open('Attendance.csv', 'r+') as f:
        mydataList = f.readlines()
        nameList = []
        for line in mydataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')


encodeListKnown = findEncodings(images)
print("Encoding Complete...")


cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    imgSmall = cv2.resize(img,(0,0),None,0.25,0.25)
    imgSmall = cv2.cvtColor(imgSmall,cv2.COLOR_BGR2RGB)

    face_curr_frame = face_recognition.face_locations(imgSmall)
    encode_curr_frame = face_recognition.face_encodings(imgSmall, face_curr_frame)

    for encodeFace, faceLoc in zip(encode_curr_frame, face_curr_frame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDist = face_recognition.face_distance(encodeListKnown, encodeFace)

        # print(faceDist)

        matchIndex = np.argmin(faceDist)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            # print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 =  y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)

