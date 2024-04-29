from django.http import StreamingHttpResponse
from django.shortcuts import render
import cv2
import threading
import time
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np


class Camera:
    def __init__(self):
        self.frame = None
        self.is_running = False
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.daemon = True

    def start(self):
        if not self.is_running:
            self.is_running = True
            self.thread.start()

    def stop(self):
        self.is_running = False
        self.thread.join()

    def _capture_frames(self):
        cap = cv2.VideoCapture(0)
        while self.is_running:
            ret, frame = cap.read()
            if not ret:
                break
            self.frame = frame
            time.sleep(0.03)
        cap.release()

# Initialize HandDetector and Classifier
detector = HandDetector(maxHands=2)
model_path = r"\Desktop\hackathon\project\app\models\keras_model.h5"
labels_path = r"\Desktop\hackathon\project\app\models\labels.txt"
try:
    classifier = Classifier(model_path, labels_path)
except FileNotFoundError as e:
    print(f"Error: {e}")

offset = 20
imgSize = 300
labels = ["Hello", "Good", "Morning", "Help", "Thankyou"]

prev_gesture = None  # Variable to store the previous gesture

def frame_generator():
    global prev_gesture  # Use the global variable for tracking previous gesture
    while camera.is_running:
        if camera.frame is not None:
            frame = camera.frame.copy()
            hands, _ = detector.findHands(frame)
            if hands:
                for hand in hands:
                    x, y, w, h = hand['bbox']
                    imgCrop = frame[y-offset:y + h + offset, x-offset:x + w + offset]
                    if not imgCrop.size == 0:
                        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
                        imgResize = cv2.resize(imgCrop, (imgSize, imgSize))
                        imgWhite[:imgSize, :imgSize] = imgResize
                        try:
                            prediction, index = classifier.getPrediction(imgWhite, draw=False)
                            label_text = labels[index]
                            cv2.rectangle(frame,(x-offset,y-offset-70),(x -offset+400, y - offset+60-50),(0,255,0),cv2.FILLED)  
                            cv2.putText(frame,label_text,(x,y-30),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2) 
                            cv2.rectangle(frame,(x-offset,y-offset),(x + w + offset, y+h + offset),(0,255,0),4)

                            
                            if label_text != prev_gesture:
                                print("Detected gesture:", label_text) 
                                prev_gesture = label_text 
                                
                        except:
                            pass
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

camera = Camera()

def camera_feed(request):
    camera.start()
    return StreamingHttpResponse(frame_generator(), content_type="multipart/x-mixed-replace;boundary=frame")

def index(request):
    return render(request, 'index.html')
