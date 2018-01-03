import numpy as np
import cv2
import time
import helpers as h
import pickle
import os
from sklearn import svm

# Step 1: Gradient

def computeGradient(img, kx, ky):

	img_x = cv2.sepFilter2D(img,cv2.CV_32F,kx, ky)
	img_y = cv2.sepFilter2D(img,cv2.CV_32F,ky, kx)
	mag, angle = cv2.cartToPolar(img_x, img_y,angleInDegrees=True)
	angle = angle%180
	del img_x
	del img_y

	magnitud = np.apply_along_axis(np.max, 2, mag)

	#t0 = time.perf_counter()
	#indices = np.apply_along_axis(np.argmax, 2, mag)
	#angulo = np.zeros(img.shape[:2])
	#for i in range(img.shape[0]):
	#	for j in range(img.shape[1]):
	#		angulo[i,j] = angle[i,j,indices[i,j]]
	#t1 = time.perf_counter()

	coord_0 = np.repeat(np.arange(img.shape[0]),img.shape[1])
	coord_1 = np.tile(np.arange(img.shape[1]),img.shape[0])

	#t3 = time.perf_counter()
	indices = np.apply_along_axis(np.argmax, 2, mag)
	angulo = angle[coord_0, coord_1, indices.flatten()]
	angulo = angulo.reshape(img.shape[:2])
	#t4 = time.perf_counter()

	del mag
	del angle

	#print("Bucle: %.3f seconds" % (t1-t0))
	#print("Otro : %.3f seconds" % (t4-t3))

	#print(np.all(angulo==cosa))
	return magnitud, angulo

# Step 2: Cell Histograms


def cellHistogram(cell_m, cell_o, bins = 9, max_angle = 180):
	bin_size = max_angle//bins
	histogram = np.zeros(bins)

	lower_bin = (cell_o//bin_size).astype(np.uint8)
	upper_bin = (lower_bin+1)%bins

	value_to_upper = (cell_o%bin_size) / bin_size
	value_to_lower = (1-value_to_upper) * cell_m
	value_to_upper *= cell_m

	for i in range(8):
		for j in range(8):
			histogram[lower_bin[i,j]] += value_to_lower[i,j]
			histogram[upper_bin[i,j]] += value_to_upper[i,j]
	
	return histogram

def computeCellHistograms(border_size, magnitud, angulo, cell_size = 8, bins =9, max_angle = 180):

	end_r = magnitud.shape[0] - border_size
	end_c = magnitud.shape[1] - border_size

	rows = magnitud.shape[0] - (border_size*2)
	cols = magnitud.shape[1] - (border_size*2)

	histograms = np.zeros((rows//cell_size, cols//cell_size, bins), np.float)

	for i in range(border_size,end_r,cell_size):
		for j in range(border_size,end_c,cell_size):
			histograms[(i-border_size)//cell_size, (j-border_size)//cell_size] = \
				cellHistogram(magnitud[i:i+cell_size, j:j+cell_size],
					angulo[i:i+cell_size,j:j+cell_size], bins, max_angle)

	return histograms

# Step 3: Normalization

def normalizeHistogram(histograms, block_size = 2, overlapping = 0.5):
	step = int(block_size*(1-overlapping))

	normalized = []
	for i in range(0,histograms.shape[0]-block_size+1, step):
		for j in range(0,histograms.shape[1]-block_size+1, step):
			norm = np.linalg.norm(histograms[i:i+block_size, j:j+block_size])
			if norm == 0:
				normalized.extend((histograms[i:i+block_size, j:j+block_size]).flatten())
			else:
				normalized.extend((histograms[i:i+block_size, j:j+block_size]/norm).flatten())
	return np.asarray(normalized)


# Warning: Esta función escribe en disco
def obtainTrainingData():
	ky = np.asarray([[0],[1],[0]])
	kx = np.asarray([[-1],[0],[1]])

	ruta_pos = "/home/laura/Documentos/VC/trabajo_final/INRIAPerson/96X160H96/Train/pos"
	ruta_neg = "/home/laura/Documentos/VC/trabajo_final/INRIAPerson/96X160H96/Train/neg"
	
	
	directorio_pos = os.listdir(ruta_pos) 
	directorio_neg = os.listdir(ruta_neg) 

	long_descriptor = (15*7)*(4*9)
	caracteristicas = np.zeros((len(directorio_pos) + len(directorio_neg), long_descriptor))
	etiquetas = np.repeat(np.asarray([1,0]), [len(directorio_pos),len(directorio_neg)])
	i = 0

	for name in directorio_pos:
		img = cv2.imread(os.path.join(ruta_pos, name))

		magnitud, angulo = computeGradient(img, kx, ky)
	
		border_size = (img.shape[0]-128) // 2
		histograms = computeCellHistograms(border_size, magnitud, angulo, 8)
		descriptor = normalizeHistogram(histograms)
		caracteristicas[i,:]=descriptor
		h.printProgressBar(i, caracteristicas.shape[0])
		i+=1


	for name in directorio_neg:
		img = cv2.imread(os.path.join(ruta_neg, name))

		magnitud, angulo = computeGradient(img, kx, ky)
	
		border_size = (img.shape[0]-128) // 2
		histograms = computeCellHistograms(border_size, magnitud, angulo, 8)
		descriptor = normalizeHistogram(histograms)
		caracteristicas[i,:]=descriptor
		i+=1
		h.printProgressBar(i, caracteristicas.shape[0])

	pickle.dump(etiquetas, open("etiquetas.pk", "wb"))
	pickle.dump(caracteristicas, open("caracteristicas.pk", "wb"))

def main():
	etiquetas = pickle.load(open("etiquetas.pk", "rb"))
	caracteristicas = pickle.load(open("caracteristicas.pk", "rb"))

	classifier = svm.LinearSVC()
	classifier.fit(caracteristicas, etiquetas)
	predicciones = classifier.predict(caracteristicas)

	aciertos = np.sum(np.equal(predicciones,etiquetas))
	print("Porcentaje de acierto Train: %.4f" % (aciertos/predicciones.shape[0]))
	ruta_neg = "/home/laura/Documentos/VC/trabajo_final/INRIAPerson/96X160H96/Train/neg"
	directorio_neg = os.listdir(ruta_neg) 


# WARNING: escribe en disco
def extractHardExamples(ruta, clasificador):
	for name in os.listdir(ruta):
		img = cv2.imread(os.path.join(ruta, name))
		for 
		for i in range(10):
			f = random.randint(0,img.shape[0]-128)
			c = random.randint(0,img.shape[1]-64)
			cv2.imwrite("training_negative_window/"+str(i)+"_"+name, img[f:f+128,c:c+64])

if __name__ == "__main__":
    main()