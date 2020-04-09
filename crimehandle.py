from tkinter import *
from tkinter import messagebox
import os
import subprocess
import time
top = Tk()

def fire():
    try:
        file = r'python firenetThroghCam.py models/test.mp4'
        os.system(file)
    except:
        print('Their is a Problem in a Path!')

def gunknife():
    try:
        file = r'python ..\alarm.py'
        os.system(file)
    except:
        print('Their is a Problem in a Path!')

def faceadd():
    try:
        file = r'python face.py'
        os.system(file)
    except:
        print('Their is a Problem in a Path!')


bac_img=PhotoImage(file='admin.png')
w = Label(top,image=bac_img)
w.pack()

l1 = Label(top,text='Crime Scene Detection \n Using ML',bd=6)
l1.config(font=("Courier", 40))
l1.place(x=50,y=20)

face_label = Label(top,text=" Face \n Detect")
face_label.config(font=("Courier", 20))
face_label.place(x=50,y=350)

face_image =  PhotoImage(file = "face.png")
b1 = Button(top, image=face_image,command=faceadd)
b1.place(x=80,y=300)

gunknife_label = Label(top,text="Gun\n&\n Knife ")
gunknife_label.config(font=("Courier", 20))
gunknife_label.place(x=300,y=350)

gunknife_image = PhotoImage(file = "gun.png")
b3 = Button(top,image=gunknife_image,command=gunknife)
b3.place(x = 330,y=300)

fire_label = Label(top,text="Fire \n Detect")
fire_label.config(font=("Courier", 20))
fire_label.place(x=550,y=350)

fire_image = PhotoImage(file = "fire.png")
b4 = Button(top,image = fire_image,command=fire)
b4.place(x = 580,y=300)
top.mainloop()