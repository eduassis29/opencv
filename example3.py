import cv2
import numpy as np

# Função para determinar se a mão está aberta ou fechada
def hand_open_or_closed(frame, prev_frame):
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
    
    # Calcula o contorno convexo da mão
    hull = cv2.convexHull(max_contour, returnPoints=False)
    
    # Encontra os defeitos de convexidade
    defects = cv2.convexityDefects(max_contour, hull)
    
    # Conta o número de defeitos de convexidade
    if defects is not None:
        count_defects = 0
        for i in range(defects.shape[0]):
            s, e, f, _ = defects[i, 0]
            start = tuple(max_contour[s][0])
            end = tuple(max_contour[e][0])
            far = tuple(max_contour[f][0])
            
            # Calcula os ângulos entre os pontos
            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            angle = (np.arccos((b**2 + c**2 - a**2) / (2*b*c)) * 180) / np.pi
            
            # Se o ângulo é menor que 90 graus, conta como um defeito de convexidade
            if angle <= 90:
                count_defects += 1
        
        # Determina se a mão está aberta ou fechada com base no número de defeitos de convexidade
        if count_defects == 0:
            return "Fechada"
        elif count_defects <= 3:
            return "Aberta"
        else:
            return "Indeterminado"

# Inicializa a captura de vídeo
cap = cv2.VideoCapture(0)

# Lê o primeiro frame
_, prev_frame = cap.read()

while True:
    # Lê o frame atual
    ret, frame = cap.read()
    if not ret:
        break
    
    # Determina se a mão está aberta ou fechada
    hand_status = hand_open_or_closed(frame, prev_frame)
    
    # Exibe o resultado na tela
    cv2.putText(frame, hand_status, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Mostra o frame com o status da mão
    cv2.imshow('Hand Status', frame)
    
    # Atualiza o frame anterior
    prev_frame = frame.copy()
    
    # Sai do loop quando 'q' é pressionado
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a captura de vídeo e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()
