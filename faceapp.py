import cv2
from random import randrange

#this loads some pre trainded data on frontal faces in opencv
trained_face_data=cv2.CascadeClassifier('C:\\Users\\user\\Desktop\\Python\\tensorEnv\\facedetection real time\\haarcascade_frontalface_default.xml')

#camera(the default one)
webcam=cv2.VideoCapture(0)

while True:
    successful_frame_read,frame=webcam.read()

        #convert the image in grey scale
    grayscale_img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
        #draw
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)
    cv2.imshow('alienc0de', frame)
    cv2.waitKey(1)





