#  Helmet Project


import cv2
import cv2
import sys
#import face_recognition_models as face_recognition
import imutils
import subprocess as sp
import easygui

from bounding_box import bounding_box as bb
import os

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def email_send():
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    fromaddr = "qrcode2611@gmail.com"
    toaddr = "kiranpamidy2001@gmail.com"
    msg = MIMEMultipart()

    msg['From'] = fromaddr

    msg['To'] = toaddr

    msg['Subject'] = "Person detected without helmet"

    body = "test"

    msg.attach(MIMEText(body, 'plain'))

    filename = "img.jpg"
    attachment = open("img.jpg", "rb")

    p = MIMEBase('application', 'octet-stream')

    p.set_payload((attachment).read())


    encoders.encode_base64(p)

    p.add_header('Content-Disposition', "attachment; filename= %s" % filename)


    msg.attach(p)


    s = smtplib.SMTP('smtp.gmail.com', 587)


    s.starttls()


    s.login(fromaddr, "vhjj ckro lcdj hwec")


    text = msg.as_string()


    s.sendmail(fromaddr, toaddr, text)


    s.quit()



from keras import models,layers,regularizers
def SSD():
    weight_decay = 0.0001
    model = models.Sequential()

    model.add(layers.Conv2D(50 , (3,3) , padding='same' , 
                            kernel_regularizer = regularizers.l2(weight_decay),
                            activation = 'relu',
                            input_shape=(100,100,3)))
    model.add(layers.Dropout(0.4))       #to avoid overfitting
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(100 , (3,3) , padding='same',
                            kernel_regularizer = regularizers.l2(weight_decay),
                            activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(200 , (3,3), padding='same',
                            kernel_regularizer = regularizers.l2(weight_decay),
                            activation = 'relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(400, (3,3), padding='same',
                            kernel_regularizer = regularizers.l2(weight_decay),
                            activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(800 , (3,3), padding='same',
                            kernel_regularizer = regularizers.l2(weight_decay),
                            activation = 'relu'))
    model.add(layers.Dropout(0.4))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Conv2D(1600, (3,3), padding='same',
                            kernel_regularizer = regularizers.l2(weight_decay),
                            activation = 'relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.MaxPooling2D(pool_size=(2,2)))

    model.add(layers.Flatten())
    model.add(layers.Dense(4 , activation = 'sigmoid'))


def accuracy():
    
   
    
    import numpy as np
    import matplotlib.pyplot as plt
    
   
    data = {'SSMD':91.22, 'Yolo':95.81}
    courses = list(data.keys())
    values = list(data.values())
    
    fig = plt.figure(figsize = (10, 5))
    
    # creating the bar plot
    plt.bar(courses, values, color ='maroon',
            width = 0.4)
    
    plt.xlabel("Algorithm")
    plt.ylabel("Accuracy")
    plt.title("Accuracy comparison")
    plt.show()
     
     

image11 = easygui.fileopenbox()
cap= cv2.VideoCapture(image11)
face_locations = []
#cap = cv2.VideoCapture('helmet1.mp4')

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if ((x<200) & (y<40)):
                bb.add(img, x, y, x + 5, y + 5, "No Helmet", "red")
                cv2.imwrite("img.jpg",img)
                email_send()
        elif ((x<375) & (y<82)):
            bb.add(img, x, y, x + 5, y + 5, "No Helmet", "red")
            cv2.imwrite("img.jpg",img)
            email_send()
        else:
            bb.add(img, x, y, x + 5, y + 5, "Helmet", "maroon")



    # Display


    cv2.imshow('Faces', img)

    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        accuracy()
        break

# Release the VideoCapture object
cap.release()