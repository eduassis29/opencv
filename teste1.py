import cv2
import numpy as np

cap = cv2.VideoCapture(0)
frame = cap.read()
cv2.imshow(frame)