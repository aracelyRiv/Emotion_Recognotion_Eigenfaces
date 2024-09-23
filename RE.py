import cv2
import os
import numpy as np

# Cargar el modelo entrenado
method = 'EigenFaces'
emotion_recognizer = cv2.face.EigenFaceRecognizer_create()
emotion_recognizer.read('../RF/modeloEigenFaces.xml')

# Ruta donde hayas almacenado las imágenes de entrenamiento
dataPath = '../RF/test'
imagePaths = os.listdir(dataPath)

# Inicializa la cámara
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Asegúrate de que el índice sea correcto

# Cargamos el clasificador de rostros
faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convertimos a escala de grises
    auxFrame = gray.copy()

    # Detectamos los rostros en la imagen
    faces = faceClassif.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Extraemos el área del rostro
        rostro = auxFrame[y:y+h, x:x+w]

        # Redimensionamos el rostro a 48x48 (tamaño usado en FER)
        rostro = cv2.resize(rostro, (48, 48), interpolation=cv2.INTER_CUBIC)

        # Predicción de la emoción
        result = emotion_recognizer.predict(rostro)
        cv2.putText(frame,'{}'.format(result),(x,y-5),1,1.3,(255,255,0),1,cv2.LINE_AA)
        # Si la predicción es confiable, mostramos el nombre de la emoción
        if result[1] < 5700:  # Ajusta este valor según tus resultados
            cv2.putText(frame, '{}'.format(imagePaths[result[0]]), (x, y-25), 2, 1.1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No identificado', (x, y-20), 2, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # Mostramos el frame con la predicción
    cv2.imshow('Emotion Recognition', frame)

    # Presionar 'Esc' para salir
    k = cv2.waitKey(1)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
