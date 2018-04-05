# import the necessary packages
from __future__ import print_function
from imutils.object_detection import non_max_suppression
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import cv2
import time
import sys


#fgbg = cv2.createBackgroundSubtractorMOG2()
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=False).start()
time.sleep(2.0)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    frame = vs.read()
    cascade_path = "haarcascade_frontalface_default.xml"
    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascade_path)


    image1 = imutils.resize(frame, height=500)
    gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30, 30),flags=0)

    if format(len(faces)) == 1:
        print("Found {0} face!".format(len(faces)))
    else:
        print("Found {0} faces!".format(len(faces)))

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("window", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


cv2.destroyAllWindows()
vs.stop()