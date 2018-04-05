from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2
from matplotlib import pyplot as plt
from gender_detection import gender_classi

len_male = 172.8
len_female = 160
len_human = 168
focal_len = 1030 # in px
GENDER = {0:'Male','1':"Female"}
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

model_path = "MobileNetSSD_deploy.caffemodel"
prototxt_path = "MobileNetSSD_deploy.prototxt.txt"

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
 
	# grab the frame dimensions and convert it to a blob
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
		0.007843, (300, 300), 127.5)
 
	# pass the blob through the network and obtain the detections and
	# predictions
	net.setInput(blob)
	detections = net.forward()
	#print (detections)

	for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections[0, 0, i, 2]
 
		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
		if confidence > 0.2:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections[0, 0, i, 1])
			if idx == 15:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				gen_image = frame[startX:endX, startY:endY]
				gen = gender_classi(gen_image)
				
				#these are coordinates of the box
				len_image = abs(endY-startY)*1.8
				depth = ((len_human*focal_len)/len_image)/30
				print depth

				X = (startX+endX)/2
				Y = (startY+endY)/2
				x_ = (depth*X)/focal_len
				y_ = (depth*Y)/focal_len
				print (x_,y_)
				X,Y=[5,x_],[0,y_]
				plt.xlim(0,10)
				plt.ylim(0,10)
				plt.scatter(X,Y)
				plt.pause(0.05)
 
			# draw the prediction on the frame
			label = CLASSES[idx]
			gender = GENDER[np.argmax(gen)]
			if label == "person":
				cv2.rectangle(frame, (startX, startY), (endX, endY),COLORS[idx], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, gender, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
 
	# update the FPS counter
	fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
 
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
 