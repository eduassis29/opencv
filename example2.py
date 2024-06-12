import cv2
import numpy as np

# Função para detectar o movimento da mão
def detect_hand_motion(frame, prev_frame):
    # Converte os frames em escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Calcula a diferença absoluta entre os frames
    diff = cv2.absdiff(gray, prev_gray)
    
    # Aplica um limiar para obter os pixels com diferença significativa
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Encontra os contornos na imagem binarizada
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Encontra o contorno da mão com a maior área
    max_contour = max(contours, key=cv2.contourArea)
    
    # Calcula o centro da mão
    moments = cv2.moments(max_contour)
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        return (cx, cy)
    else:
        return None

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)

# Lê o primeiro frame
_, prev_frame = cap.read()

while True:
    # Lê o frame atual
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detecta o movimento da mão
    hand_pos = detect_hand_motion(frame, prev_frame)
    
    # Desenha um círculo no centro da mão, se detectada
    if hand_pos is not None:
        cv2.circle(frame, hand_pos, 10, (0, 255, 0), -1)
    
    # Mostra o frame com a detecção de movimento
    cv2.imshow('Hand Motion Detection', frame)
    
    # Atualiza o frame anterior
    prev_frame = frame.copy()
    
    # Sai do loop quando 'q' é pressionado
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura de vídeo e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()
