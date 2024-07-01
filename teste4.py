import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

def draw_line(image, point1, point2, color, thickness):
    cv2.line(image, point1, point2, color, thickness)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)

                ret, img = cap.read()

                if not ret:
                    break
                
                # Convert the img to RGB for processing with Mediapipe
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Process the img with Mediapipe Hands to find hand landmarks
                results = hands.process(img_rgb)
                
                if results.multi_hand_landmarks:
                    # Assume we are tracking the first hand detected
                    hand_landmarks = results.multi_hand_landmarks[0]
                    
                    # Extract landmark coordinates (you may need to adjust the landmarks based on your needs)
                    landmarks = hand_landmarks.landmark
                    # For example, get the coordinates of index finger tip and thumb tip
                    if len(landmarks) >= 8:  # Assuming at least 8 landmarks are detected
                        x1 = int(landmarks[mpHands.HandLandmark.INDEX_FINGER_TIP].x * img.shape[1])
                        y1 = int(landmarks[mpHands.HandLandmark.INDEX_FINGER_TIP].y * img.shape[0])
                        x2 = int(landmarks[mpHands.HandLandmark.THUMB_TIP].x * img.shape[1])
                        y2 = int(landmarks[mpHands.HandLandmark.THUMB_TIP].y * img.shape[0])
                        
                        # Draw the line between index finger tip and thumb tip
                        draw_line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                

        #mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break