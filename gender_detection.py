import numpy as np
import argparse
import imutils
import time
import cv2

CLASSES = ["MALE","FEMALE"]
model_path = "gender_net.caffemodel"
prototxt_path = "deploy_gender.prototxt.txt"

def gender_classi(image):

	net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (227,227)))

	net.setInput(blob)
	detections = net.forward()

	return np.argmax(detections)