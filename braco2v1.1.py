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

i_lmy = 0
j_lmy = 0

tip = {8:0, 12:0, 16:0, 20:0}
mcp = {5:0, 9:0, 13:0, 17:0}
contador_c = [0, 0, 0, 0]

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

                c = 0
                for i,j in zip(tip, mcp):
                    if id == i:
                        i_lmy = lm.y
                        
                        
                    if id == j:
                        j_lmy = lm.y

                    if i_lmy > j_lmy:
                        tip[i] = 1
                    
                    else:
                        tip[i] = 0
                    
                print(tip)
                

                if sum(tip.values()) == 4:
                    cv2.putText(img,str('Mao Aberta'), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

                elif sum(tip.values()) == 0:
                    cv2.putText(img,str('Mao Fechada'), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
                
                else:
                    cv2.putText(img,str('Mao Fora de Padrao'), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

                
                
                if id ==0:
                    cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)
                
        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        # time.sleep(1)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    #cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break