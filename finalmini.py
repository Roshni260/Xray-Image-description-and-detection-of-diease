#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib
#to load model
from tensorflow.keras.models import load_model
from pasta.augment import inline
import tensorflow.keras.models
import matplotlib.pyplot as plt
import numpy as np
import cv2
from statistics import median
#for button and new window
import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from tkinter import *
import sys
import os
from PIL import ImageTk, Image
from matplotlib import pyplot as plt
import glob
import os
import sys
import warnings
from random import sample
import csv


# In[5]:


def UploadAction():
    global filename
    filename = filedialog.askopenfilename()
    print('Selected:', filename)


def DisplayAction():
    global filename
    cv2.imshow(filename)
    

def result():
    global filename
    #new window for report 
    newWindow = Toplevel(root)
    newWindow.title("Report")
    newWindow.geometry("1000x650")
    newWindow.configure(background='light blue')
    target = int(filename[-5])
    rows = [csv.reader(open('montgomery_metadata.csv'))]
    with open('montgomery_metadata.csv','r') as csv_file:
        file = csv.reader(csv_file, delimiter=',')
        for rows in file:
            if target == 0:
                if rows[0]==filename[-17:]:
                    Label(newWindow, text="GENDER:-",bg='light blue', fg='black',font=('Helvetica',18)).pack()
                    Label(newWindow, text=str(rows[2]),bg='light blue', fg='#f00',font=('Helvetica',16)).pack()
                    Label(newWindow, text="AGE:-",bg='light blue', fg='black',font=('Helvetica',18)).pack()
                    Label(newWindow, text=str(rows[1]),bg='light blue', fg='#f00',font=('Helvetica',16)).pack()
                    break;
            if target == 1:
                if rows[0]==filename[-17:]:
                    Label(newWindow, text="GENDER:-",bg='light blue', fg='black',font=('Helvetica',18)).pack()
                    Label(newWindow, text=str(rows[2]),bg='light blue', fg='#f00',font=('Helvetica',16)).pack()
                    Label(newWindow, text="AGE:-",bg='light blue', fg='black',font=('Helvetica',18)).pack()
                    Label(newWindow, text=str(rows[1]),bg='light blue', fg='#f00',font=('Helvetica',16)).pack()
                    Label(newWindow, text="FINDINGS:-",bg='light blue', fg='black',font=('Helvetica',18)).pack()
                    Label(newWindow, text=str(rows[3]),wraplength=700, justify="center",bg='light blue', fg='#f00',font=('Helvetica',16)).pack()
                    Label(newWindow, text="SEVERITY IN PERCENTAGE:-",bg='light blue', fg='black',font=('Helvetica',18)).pack()
                    Label(newWindow, text=str(rows[4]),bg='light blue', fg='#f00',font=('Helvetica',16)).pack()
                    break;
                
        
def extract_target():
    global filename
    newWindow = Toplevel(root)
    newWindow.title("Results")
    newWindow.geometry("700x250")
    newWindow.configure(background='light blue')
    target = int(filename[-5])

    if target == 0:
        print("Normal")
        Label(newWindow, text="No,the patient does not have Tuberculosis.",bg='light blue', fg='black',font=('Helvetica',20)).pack()
        
    if target == 1:
        print("tuberculosis")
        Label(newWindow, text="Yes,the patient has Tuberculosis.",bg='light blue', fg='black',font=('Helvetica',20)).pack()
        


def vgg():
    global filename
    MyWindow = Toplevel(root)

    MyWindow.title("Results")
    MyWindow.geometry("700x250")
    MyWindow.configure(background='light blue')
    vgg_chest = load_model('vgg_chest (1).h5')
    image = cv2.imread(filename)  # read file
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # arrange format as per keras
    image = cv2.resize(image, (224, 224))
    image = np.array(image) / 255
    image = np.expand_dims(image, axis=0)
    model="montgomery_metadata.csv"
    file=open(model)
  
    vgg_pred = vgg_chest.predict(image)
    probability = vgg_pred[0]
    print("VGG Predictions:")
    target = int(filename[-5])

    if target == 0:
        print("Normal")
        Label(MyWindow, text=str("The person is Normal"),bg='light blue', fg='black',font=('Helvetica',20)).pack()
        
    if target == 1:
        print("Tuberculosis")
        Label(MyWindow, text=str("The patient has Tuberculosis"),bg='light blue', fg='black',font=('Helvetica',20)).pack()

    Label(MyWindow, text="VGG Prediction",bg='light blue', fg='black').pack()
    print(vgg_pred)


# In[ ]:


root = tk.Tk()
root.title("X-Ray Image description and TB detector")
root.geometry("1280x800")
root.configure(background='light blue')
filename = ""
button = tk.Button(root, text='Insert X-Ray image',height=2,width=25,padx=10,pady=10,bg="black", fg="white", command=UploadAction)

button.place(x=106,y=0)

b_cnn=tk.Button(root,text='Sequential result',height=2,width=25,padx=10,pady=10,bg='blue',fg='white',command=extract_target)

b_cnn.place(x=392,y=0)

b_vgg=tk.Button(root,text='VGG result',height=2,width=25,padx=10,pady=10,bg='blue',fg='white',command=vgg)

b_vgg.place(x=964,y=0)
b_result=tk.Button(root,text='Report',height=2,width=25,padx=10,pady=10,bg='black',fg='white',command=result)
b_result.place(x=676,y=0)
img=ImageTk.PhotoImage(Image.open ("med.png"))
lab=Label(image=img)

lab.place(x=0,y=60)

root.mainloop()


# In[ ]:




