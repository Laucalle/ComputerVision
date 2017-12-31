import numpy as np
import cv2
import prt1 as p
import time

img = cv2.imread("crop001029a.png")

ky = np.asarray([[0],[1],[0]])
kx = np.asarray([[-1],[0],[1]])

img_x = cv2.sepFilter2D(img,cv2.CV_32F,kx, ky)
img_y = cv2.sepFilter2D(img,cv2.CV_32F,ky, kx)
mag, angle = cv2.cartToPolar(img_x, img_y,angleInDegrees=True)
angle = angle%180
del img_x
del img_y

magnitud = np.apply_along_axis(np.max, 2, mag)

t0 = time.perf_counter()
indices = np.apply_along_axis(np.argmax, 2, mag)
angulo = np.zeros(img.shape[:2])
for i in range(img.shape[0]):
	for j in range(img.shape[1]):
		angulo[i,j] = angle[i,j,indices[i,j]]
t1 = time.perf_counter()

coord_0 = np.repeat(np.arange(img.shape[0]),img.shape[1])
coord_1 = np.tile(np.arange(img.shape[1]),img.shape[0])

t3 = time.perf_counter()
indices = np.apply_along_axis(np.argmax, 2, mag)
cosa = angle[coord_0, coord_1, indices.flatten()]
cosa = cosa.reshape(img.shape[:2])
t4 = time.perf_counter()

del mag
del angle

print("Bucle: %.3f seconds" % (t1-t0))
print("Otro : %.3f seconds" % (t4-t3))

print(np.all(angulo==cosa))

border_size = (img.shape[0]-128) // 2
borderless = img[border_size:-border_size, border_size:-border_size]

def cellHistogram(cell_m, cell_o):
	bins = 9
	bin_size = 20
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


histogramas = []
for i in range(0,borderless.shape[0],8):
	for j in range(0,borderless.shape[1],8):
		histogramas.append(cellHistogram(magnitud[i:i+8, j:j+8], angulo[i:i+8,j:j+8]))

