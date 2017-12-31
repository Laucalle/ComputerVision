import numpy as np
import cv2
import time
import helpers as h


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
			print("%d %d" % (i,j))
			norm = np.linalg.norm(histograms[i:i+block_size, j:j+block_size])
			normalized.extend((histograms[i:i+block_size, j:j+block_size]/norm).flatten())
	return normalized

def main():
	img = cv2.imread("crop001029a.png")
	ky = np.asarray([[0],[1],[0]])
	kx = np.asarray([[-1],[0],[1]])
	magnitud, angulo = computeGradient(img, kx, ky)
	
	border_size = (img.shape[0]-128) // 2
	histograms = computeCellHistograms(border_size, magnitud, angulo, 8)
	descriptor = normalizeHistogram(histograms)

if __name__ == "__main__":
    main()