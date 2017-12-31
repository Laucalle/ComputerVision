import numpy as np
import cv2
from math import sqrt, ceil, floor, exp
from matplotlib import pyplot as plt
from itertools import repeat
def showMatrix(img, labels, grey = True, col = None, row= None):
	# Para disponer las imagenes en una matriz cuadrada
	if col is None:
		col = ceil(sqrt(len(img)))
	if row is None:
		row = ceil(sqrt(len(img)))
	for i in range(0,len(img)):
		# seleccionamos que lugar vamos a usar en la matriz
		plt.subplot(row,col,i+1)
		plt.title(labels[i])
		plt.xticks([]), plt.yticks([]) # elimina los ejes
		# imprime la imagen invirtiendo el orden de los colores:
		# openCV utiliza RGB y matplotlib BGR
		if grey :
			if len(img[i].shape)>2:
				plt.imshow(img[i][:,:,0], cmap = 'gray')
			else:
				plt.imshow(img[i][:,:], cmap = 'gray')
		else:
			if len(img[i].shape)>2:
				plt.imshow(img[i][:,:,::-1])
			else:	
				plt.imshow(img[i])
	plt.show()


# esta funcion transforma los valores al intervalo 0-255
def CenterScale(img):
	return np.uint8(((img-img.min()*1.0)/(img.max()-img.min()))*255)
