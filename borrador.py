import numpy as np
import cv2
import time
import helpers as h
import pickle
import os
from sklearn import svm

# Step 1: Gradient

def computeGradient(img, kx, ky):

	img_x = cv2.sepFilter2D(img, cv2.CV_32F, kx, ky, borderType=cv2.BORDER_REPLICATE)
	img_y = cv2.sepFilter2D(img, cv2.CV_32F, ky, kx, borderType=cv2.BORDER_REPLICATE)
	mag, angle = cv2.cartToPolar(img_x, img_y,angleInDegrees=True)
	angle = angle%180
	del img_x
	del img_y

	magnitudes = np.apply_along_axis(np.max, 2, mag)

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
	angles = angle[coord_0, coord_1, indices.flatten()]
	angles = angulo.reshape(img.shape[:2])
	#t4 = time.perf_counter()

	del mag
	del angle

	#print("Bucle: %.3f seconds" % (t1-t0))
	#print("Otro : %.3f seconds" % (t4-t3))

	#print(np.all(angulo==cosa))
	return magnitudes, angles

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

def computeCellHistograms(border_size, magnitudes, angles, cell_size = 8, bins = 9, max_angle = 180):

	end_r = magnitudes.shape[0] - border_size
	end_c = magnitudes.shape[1] - border_size

	rows = magnitudes.shape[0] - (border_size*2)
	cols = magnitudes.shape[1] - (border_size*2)

	histograms = np.zeros((rows//cell_size, cols//cell_size, bins), np.float)

	for i in range(border_size,end_r,cell_size):
		for j in range(border_size,end_c,cell_size):
			histograms[(i-border_size)//cell_size, (j-border_size)//cell_size] = \
				cellHistogram(magnitudes[i:i+cell_size, j:j+cell_size],
					angles[i:i+cell_size,j:j+cell_size], bins, max_angle)

	return histograms

# Step 3: Normalization

def normalizeHistograms(histograms, block_size = 2, overlapping = 0.5):
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

def obtainTrainingData(deriv_kernel, pos_path, neg_path):
	kx = deriv_kernel[0]
	ky = deriv_kernel[1]

	pos_directory = os.listdir(pos_path)
	neg_directory = os.listdir(neg_path)

	descriptor_length = (15*7)*(4*9)
	features = np.zeros((len(pos_directory) + len(neg_directory), descriptor_length))
	labels = np.repeat(np.asarray([1,0]), [len(pos_directory),len(neg_directory)])
	i = 0

	for name in pos_directory:
		img = cv2.imread(os.path.join(pos_path, name))

		magnitudes, angles = computeGradient(img, kx, ky)

		border_size = (img.shape[0]-128) // 2
		histograms = computeCellHistograms(border_size, magnitudes, angles, 8)
		descriptor = normalizeHistograms(histograms)
		features[i,:] = descriptor
		h.printProgressBar(i, features.shape[0])
		i+=1

	for name in neg_directory:
		img = cv2.imread(os.path.join(neg_path, name))

		magnitudes, angles = computeGradient(img, kx, ky)

		border_size = (img.shape[0]-128) // 2
		histograms = computeCellHistograms(border_size, magnitudes, angles, 8)
		descriptor = normalizeHistograms(histograms)
		features[i,:] = descriptor
		i+=1
		h.printProgressBar(i, features.shape[0])

	return features, labels

# Warning: this function writes data to disk
def extractNegativeWindows(path, classifier):
	for name in os.listdir(path):
		img = cv2.imread(os.path.join(path, name))
		for i in range(10):
			f = random.randint(0,img.shape[0]-128)
			c = random.randint(0,img.shape[1]-64)
			cv2.imwrite("training_negative_window/"+str(i)+"_"+name, img[f:f+128,c:c+64])

def extractHardExamples(path, deriv_kernel, classifier, stride = 8):
	n_examples = 1
	for name in os.listdir(path):
		img = cv2.imread(os.path.join(path, name))
		dims = (img.shape[0]/1.2,img.shape[1]/1.2,img.shape[2])
		pyramid = [img]

		while dims[0] > 128 and dims[1] > 64:
			pyramid.append(cv2.pyrDown(src=pyramid[-1],dstsize=dims))
			dims = (int(dims[0]/1.2),int(dims[1]/1.2),dims[2])

		for level in pyramid:
			margin_r = int((dims[0] % 8) // 2)
			margin_c = int((dims[1] % 8) // 2)
			magnitudes, angles = computeGradient(level[margin_r:-margin_r,margin_c:-margin_c],*deriv_kernel)
			histograms = computeCellHistograms(0,magnitudes,angles)
		    for i in range(0,level.shape[0]-2*margin_r+1,stride):
				for j in range(0,level.shape[1]-2*margin_c+1,stride):
					window_norm_histograms = normalizeHistograms(histograms[i:i+128,j:j+64])
					if classifier.predict(window_norm_histograms)[0] > 0:
						cv2.imwrite("hard_examples/"+str(n_examples)+".png",histograms[i:i+128,j:j+64])


def main():

	pos_train_path = "/home/laura/Documentos/VC/trabajo_final/INRIAPerson/96X160H96/Train/pos"
	neg_train_path = "/home/laura/Documentos/VC/trabajo_final/INRIAPerson/96X160H96/Train/neg"

	kx = np.asarray([[-1],[0],[1]])
	ky = np.asarray([[0],[1],[0]])

	features, labels = obtainTrainingData((kx,ky),pos_train_path,neg_train_path)
	pickle.dump(labels, open("labels.pk", "wb"))
	pickle.dump(features, open("features.pk", "wb"))
	labels = pickle.load(open("labels.pk", "rb"))
	features = pickle.load(open("features.pk", "rb"))

	classifier = svm.LinearSVC()
	classifier.fit(features, labels)
	predictions = classifier.predict(features)

	correct_answers = np.sum(np.equal(predictions,labels))
	print("Training accuracy: %.4f" % (correct_answers/predictions.shape[0]))
	neg_directory = os.listdir(neg_train_path)

if __name__ == "__main__":
	main()
