# this one is used to recognize the
# face after training the model with
# our data stored using knn
import cv2
import numpy as np
import pandas as pd
#from playsound import playsound

from npwriter import f_name
from sklearn.neighbors import KNeighborsClassifier

from tkinter import *
import os
import npwriter

root=Tk()
root.geometry('480x150')
root.title('face module')

def face_recog():
	# reading the data
	data = pd.read_csv(f_name).values

	# data partition
	X, Y = data[:, 1:-1], data[:, -1]

	print(X, Y)

	# Knn function calling with k = 5
	model = KNeighborsClassifier(n_neighbors=5)

	# fdtraining of model
	model.fit(X, Y)

	cap = cv2.VideoCapture(0)

	classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

	f_list = []

	while True:

	    ret, frame = cap.read()

	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	    faces = classifier.detectMultiScale(gray, 1.5, 5)

	    X_test = []

	    # Testing data
	    for face in faces:
	        x, y, w, h = face
	        im_face = gray[y:y + h, x:x + w]
	        im_face = cv2.resize(im_face, (100, 100))
	        X_test.append(im_face.reshape(-1))

	    if len(faces) > 0:
	        response = model.predict(np.array(X_test))
	        # prediction of result using knn

	        for i, face in enumerate(faces):
	            x, y, w, h = face

	            # drawing a rectangle on the detected face
	            cv2.rectangle(frame, (x, y), (x + w, y + h),
	                          (255, 0, 0), 3)

	            # adding detected/predicted name for the face
	            cv2.putText(frame, response[i], (x - 50, y - 50),
	                        cv2.FONT_HERSHEY_DUPLEX, 2,
	                        (0, 255, 0), 3)

	    cv2.imshow("full", frame)
	    #playsound('Twin-bell-alarm-clock-sound.mp3')
	    key = cv2.waitKey(1)

	    if key & 0xFF == ord("q"):
	        break

	cap.release()
	cv2.destroyAllWindows()

def face_detect():
	name = input("Enter your name: ")

	# this is used to access the web-cam
	# in order to capture frames
	cap = cv2.VideoCapture(0)

	classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

	# this is class used to detect the faces as provided
	# with a haarcascade_frontalface_default.xml file as data
	f_list = []

	while True:
	    ret, frame = cap.read()

	    # converting the image into gray
	    # scale as it is easy for detection
	    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	    # detect multiscale, detects the face and its coordinates
	    faces = classifier.detectMultiScale(gray, 1.5, 5)

	    # this is used to detect the face which
	    # is closest to the web-cam on the first position
	    faces = sorted(faces, key=lambda x: x[2] * x[3],
	                   reverse=True)

	    # only the first detected face is used
	    faces = faces[:1]

	    # len(faces) is the number of
	    # faces showing in a frame
	    if len(faces) == 1:
	        # this is removing from tuple format
	        face = faces[0]

	        # storing the coordinates of the
	        # face in different variables
	        x, y, w, h = face

	        # this is will show the face
	        # that is being detected
	        im_face = frame[y:y + h, x:x + w]

	        cv2.imshow("face", im_face)

	    if not ret:
	        continue

	    cv2.imshow("full", frame)

	    key = cv2.waitKey(1)

	    # this will break the execution of the program
	    # on pressing 'q' and will click the frame on pressing 'c'
	    if key & 0xFF == ord('q'):
	        break
	    elif key & 0xFF == ord('c'):
	        if len(faces) == 1:
	            gray_face = cv2.cvtColor(im_face, cv2.COLOR_BGR2GRAY)
	            gray_face = cv2.resize(gray_face, (100, 100))
	            print(len(f_list), type(gray_face), gray_face.shape)

	            # this will append the face's coordinates in f_list
	            f_list.append(gray_face.reshape(-1))
	        else:
	            print("face not found")

	        # this will store the data for detected
	        # face 10 times in order to increase accuracy
	        if len(f_list) == 10:
	            break

	# declared in npwriter
	npwriter.write(name, np.array(f_list))

	cap.release()
	cv2.destroyAllWindows()

btn1=Button(root, text='Face Recognize', bd='3', height=5, width=20, font=10, command=face_recog).grid(row=5, column=1 ,padx=30, pady=10)
btn2=Button(root, text='Add New Face ', bd='3', height=5, width=20, font=10, command=face_detect).grid(row=5, column=4, pady=10)



root.resizable(0, 0)
root.mainloop()
