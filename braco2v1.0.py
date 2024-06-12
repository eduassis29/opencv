import cv2
import mediapipe as mp
import time
cap = cv2.VideoCapture(0)

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
    print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                if id == 5:
                    id_5 = lm.y
                elif id == 8:
                    id_8 = lm.y
                elif id == 9:
                    id_9 = lm.y
                elif id == 12:
                    id_12 = lm.y
                elif id == 13:
                    id_13 = lm.y
                elif id == 16:
                    id_16 = lm.y
                elif id == 17:
                    id_17 = lm.y
                elif id == 20:
                    id_20 = lm.y
                if id ==0:
                    cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)
        
            if id_5 < id_8 and id_9 < id_12 and id_13 < id_16 and id_17 < id_20:
                    mao_estado = True
                    cv2.putText(img,str('Mao Fechada'), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
                    print('Mão Fechada')
            elif id_5 > id_8 and id_9 > id_12 and id_13 > id_16 and id_17 > id_20:
                mao_estado = False
                cv2.putText(img,str('Mao Aberta'), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    #cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break