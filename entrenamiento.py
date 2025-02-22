import cv2
import os
import numpy as np

dataPath = '/Users/admin/Documents/PROJECTS/IA/camaras_ia/data'
peopleList = os.listdir(dataPath)
print('Lista de personas: ', peopleList)
peopleList = [person for person in peopleList if person != '.DS_Store']

labels = []
facesData = []
label = 0

for nameDir in peopleList:
	personPath = dataPath + '/' + nameDir
	print('Leyendo las imágenes')

	for fileName in os.listdir(personPath):
		if fileName == '.DS_Store':
			continue
		print('Rostros: ', nameDir + '/' + fileName)
		labels.append(label)
		facesData.append(cv2.imread(personPath+'/'+fileName,0))
		#image = cv2.imread(personPath+'/'+fileName,0)
	label = label + 1

# Métodos para entrenar el reconocedor
face_recognizer = cv2.face.FisherFaceRecognizer_create()

# Entrenando el reconocedor de rostros
print("Entrenando...")
face_recognizer.train(facesData, np.array(labels))

# Almacenando el modelo obtenido
face_recognizer.write('modeloFisherFace.xml')
print("Modelo almacenado...")