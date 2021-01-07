import cv2
import numpy as np
import face_recognition

imgElon = face_recognition.load_image_file('trained/elon.jpg')
imgElon = cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)

imgElonTest = face_recognition.load_image_file('trained/virat.jpg')
imgElonTest = cv2.cvtColor(imgElonTest,cv2.COLOR_BGR2RGB)

# main image
face_location = face_recognition.face_locations(imgElon)[0]
face_elon_encoding = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon,(face_location[3],face_location[0]),(face_location[1],face_location[2]),(255,0,255),2)

# test image
face_location_test = face_recognition.face_locations(imgElonTest)[0]
face_elon_encoding_test = face_recognition.face_encodings(imgElonTest)[0]
cv2.rectangle(imgElonTest,(face_location_test[3],face_location_test[0]),(face_location_test[1],face_location_test[2]),(255,0,255),2)

# comapare and recognize.
results = face_recognition.compare_faces([face_elon_encoding],face_elon_encoding_test)
face_distance = face_recognition.face_distance([face_elon_encoding],face_elon_encoding_test)

# print(results, face_distance)
cv2.putText(imgElonTest,f'{results} {round(face_distance[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)



cv2.imshow('Elon Mask',imgElon)
cv2.imshow('Elon Mask Tested',imgElonTest)
cv2.waitKey(0)
