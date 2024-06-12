import cv2

# Função para detectar a mão
def detectar_mao(imagem):
    # Converter a imagem para tons de cinza
    imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    # Aplicar um desfoque para reduzir o ruído
    imagem_desfocada = cv2.GaussianBlur(imagem_cinza, (7, 7), 0)

    # Detectar as bordas usando o Canny edge detector
    bordas = cv2.Canny(imagem_desfocada, 50, 150)

    # Encontrar os contornos na imagem
    contornos, _ = cv2.findContours(bordas.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Verificar se há contornos detectados
    if len(contornos) == 0:
        return None

    # Encontrar o maior contorno (a mão)
    mao_contorno = max(contornos, key=cv2.contourArea)

    # Definir um retângulo delimitador em torno da mão
    (x, y, largura, altura) = cv2.boundingRect(mao_contorno)

    # Retornar as coordenadas do retângulo delimitador
    return x, y, largura, altura

# Função para determinar se a mão está aberta ou fechada
def verificar_mao_aberta_fechada(imagem_mao):
    # Definir a área de interesse (região da palma da mão)
    ROI = imagem_mao[60:300, 60:300]

    # Converter a imagem para tons de cinza
    ROI_cinza = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY)

    # Aplicar um desfoque para reduzir o ruído
    ROI_desfocada = cv2.GaussianBlur(ROI_cinza, (7, 7), 0)

    # Binarizar a imagem usando uma limiarização adaptativa
    _, ROI_binaria = cv2.threshold(ROI_desfocada, 127, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # Contar o número de pixels brancos na imagem binarizada
    total_pixels_brancos = cv2.countNonZero(ROI_binaria)

    # Calcular a porcentagem de pixels brancos em relação à área total
    porcentagem_pixels_brancos = (total_pixels_brancos / (ROI_binaria.shape[0] * ROI_binaria.shape[1])) * 100

    # Determinar se a mão está aberta ou fechada com base na porcentagem de pixels brancos
    if porcentagem_pixels_brancos < 30:
        return "Fechada"
    else:
        return "Aberta"

# Carregar o classificador de detecção de faces pré-treinado
classificador_faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Inicializar a captura de vídeo
captura = cv2.VideoCapture(0)

while True:
    # Ler o próximo frame do vídeo
    _, frame = captura.read()

    # Converter o frame para tons de cinza
    frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar as faces no frame
    faces = classificador_faces.detectMultiScale(frame_cinza, 1.3, 5)

    # Verificar se há uma face detectada
    if len(faces) > 0:
        # Obter as coordenadas da primeira face detectada
        (x, y, largura, altura) = faces[0]

        # Extrair a região de interesse (mão) com base nas coordenadas da face
        mao = frame[y:y+altura, x:x+largura]

        # Detectar a mão na região de interesse
        resultado_mao = detectar_mao(mao)

        # Verificar se uma mão foi encontrada
        if resultado_mao is not None:
            # Obter as coordenadas do retângulo delimitador da mão
            (x_mao, y_mao, largura_mao, altura_mao) = resultado_mao

            # Desenhar um retângulo delimitador em torno da mão
            cv2.rectangle(frame, (x + x_mao, y + y_mao), (x + x_mao + largura_mao, y + y_mao + altura_mao), (0, 255, 0), 2)

            # Verificar se a mão está aberta ou fechada
            resultado = verificar_mao_aberta_fechada(mao)

            # Exibir o resultado na tela
            cv2.putText(frame, resultado, (x + x_mao, y + y_mao - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Exibir o frame resultante
    cv2.imshow("Detecção de Mão", frame)

    # Verificar se a tecla 'q' foi pressionada para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar os recursos
captura.release()
cv2.destroyAllWindows()