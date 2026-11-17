import cv2
from ultralytics import YOLO

print("OpenCV version:", cv2.__version__)

model = YOLO("best.pt")

cap = cv2.VideoCapture(0)
ok, frame = cap.read()

print("Webcam works:", ok)

if ok:
    results = model(frame, verbose=False)
    print("Objects seen:", [model.names[int(c)] for c in results[0].boxes.cls])

cap.release()