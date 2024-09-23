import cv2
import os
import numpy as np
import time

def obtenerModelo(facesData, labels):
    # Crear el reconocedor EigenFaces
    emotion_recognizer = cv2.face.EigenFaceRecognizer_create()

    # Entrenando el reconocedor de rostros
    print("Entrenando con EigenFaces...")
    inicio = time.time()
    emotion_recognizer.train(facesData, np.array(labels))
    tiempoEntrenamiento = time.time() - inicio
    print("Tiempo de entrenamiento (EigenFaces): ", tiempoEntrenamiento)

    # Almacenando el modelo obtenido
    emotion_recognizer.write("modeloEigenFaces.xml")

# Ruta donde hayas almacenado las imágenes de entrenamiento
dataPath = 'C:/Ciclo8/DSM/RF/test'
emotionsList = os.listdir(dataPath)
print('Lista de personas: ', emotionsList)

labels = []
facesData = []
label = 0

for nameDir in emotionsList:
    emotionsPath = os.path.join(dataPath, nameDir)

    for fileName in os.listdir(emotionsPath):
        img_path = os.path.join(emotionsPath, fileName)
        face_image = cv2.imread(img_path, 0)

        if face_image is not None:
            labels.append(label)
            facesData.append(face_image)
            print(f"Cargada: {img_path}")
        else:
            print(f"Error al cargar la imagen: {img_path}")

    label += 1

# Entrenando solo con EigenFaces
obtenerModelo(facesData, labels)
