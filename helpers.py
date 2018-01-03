import numpy as np
import cv2
from math import sqrt, ceil, floor, exp
from matplotlib import pyplot as plt
from itertools import repeat
from matplotlib import pyplot as plt
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

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
