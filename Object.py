''' import cv2
import time
import numpy as np


 
# Load the COCO class names
with open('coco.names.txt', 'r') as f:
   class_names = f.read().split('\n')

print(class_names)
 
# Get a different colors for each of the classes
colors = np.random.uniform(0, 255, size=(len(class_names), 3))
 
# Load the DNN model
model = cv2.dnn.readNet(model='frozen_inference_graph.pb', 
                        config='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt', 
                        framework='TensorFlow')

# Set backend and target to CUDA to use GPU
model.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
model.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
 
# Webcam
cap = cv2.VideoCapture(0)

min_confidence_score = 0.6


while cap.isOpened():

    # Read in the image
    success, img = cap.read()

    imgHeight, imgWidth, channels = img.shape

    # Create blob from image
    blob = cv2.dnn.blobFromImage(img, size=(300,300), mean=(104,117,123), swapRB=True)

    # start time to calculate FPS
    start = time.time()

    # Set input to the model
    model.setInput(blob)

    # Make forward pass in model
    output = model.forward()
 
    # End time
    end = time.time()

    # calculate the FPS for current frame detection
    fps = 1 / (end-start)


    # Run over each of the detections
    for detection in output[0,0,:,:]:

        confidence = detection[2]

        if confidence > min_confidence_score:

            class_id = detection[1]

            class_name = class_names[int(class_id)-1]
            color = colors[int(class_id)]

            bboxX = detection[3] * imgWidth
            bboxY = detection[4] * imgHeight

            bboxWidth = detection[5] * imgWidth
            bboxHeight = detection[6] * imgHeight

            cv2.rectangle(img, (int(bboxX), int(bboxY)), (int(bboxWidth), int(bboxHeight)),color, thickness=2)

            cv2.putText(img, class_name, (int(bboxX), int(bboxY - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


    
    # Show FPS
    cv2.putText(img, f"{fps:.2f} FPS", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('image', img)


    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

 

cap.release()
cv2.destroyAllWindows() '''



import cv2
import os
from gtts import gTTS
import pyttsx3
thres = 0.45 # Threshold to detect object

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)


classNames= []
classFile = 'coco.names.txt'
print(classFile)
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    success,img = cap.read()
    classIds, confs, bbox = net.detect(img,confThreshold=thres)
    print(classIds,bbox)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness=2)
            
            cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
            cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
            
            engine = pyttsx3.init()
            engine.say(classNames[classId-1].upper())
            engine.runAndWait()
            
            """
            output = gTTS(classNames[classId-1].upper(), lang="ar")
            engine.save("output.mp3")
            os.system("start output.mp3")
            """
    cv2.imshow("Output",img)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break