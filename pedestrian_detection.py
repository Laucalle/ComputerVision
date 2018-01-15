import numpy as np
import cv2
import time
import helpers as h
import pickle
import os
from sklearn import svm, linear_model, metrics
import random
import multiprocessing

def norm_1(x):
	norm = np.linalg.norm(x,0)
	return x/norm if norm != 0 else x

def norm_2(x):
	norm = np.linalg.norm(x)
	return x/norm if norm != 0 else x

def norm_2_hys(x):
	x = norm_2(x)
	x[x > 0.2] = 0.2
	return norm_2(x)

# Step 1: Gradient

def computeGradient(img, kx, ky):

	img_x = cv2.sepFilter2D(img, cv2.CV_32F, kx, ky,
	 						borderType=cv2.BORDER_REPLICATE)
	img_y = cv2.sepFilter2D(img, cv2.CV_32F, ky, kx,
							borderType=cv2.BORDER_REPLICATE)

	mag, angle = cv2.cartToPolar(img_x, img_y,angleInDegrees=True)
	angle = angle%180
	del img_x
	del img_y

	magnitudes = np.apply_along_axis(np.max, 2, mag)

	coord_0 = np.repeat(np.arange(img.shape[0]),img.shape[1])
	coord_1 = np.tile(np.arange(img.shape[1]),img.shape[0])

	indices = np.apply_along_axis(np.argmax, 2, mag)
	angles = angle[coord_0, coord_1, indices.flatten()]
	angles = angles.reshape(img.shape[:2])

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

def computeCellHistograms(border_s, magnitudes, angles, cell_s = 8,
 							bins = 9, max_angle = 180):

	end_r = magnitudes.shape[0] - border_s
	end_c = magnitudes.shape[1] - border_s

	rows = magnitudes.shape[0] - (border_s*2)
	cols = magnitudes.shape[1] - (border_s*2)

	histograms = np.zeros((rows//cell_s, cols//cell_s, bins), np.float)

	for i in range(border_s,end_r,cell_s):
		for j in range(border_s,end_c,cell_s):
			# pixel coordinates to cell coordinates
			histograms[(i-border_s)//cell_s, (j-border_s)//cell_s] = \
				cellHistogram(magnitudes[i:i+cell_s, j:j+cell_s],
					angles[i:i+cell_s,j:j+cell_s], bins, max_angle)

	return histograms

# Step 3: Normalization

def normalizeHistograms(histograms, norm_f = norm_2, block_size = 2,
													 overlapping = 0.5):
	step = int(block_size*(1-overlapping))
	normalized = []
	for i in range(0,histograms.shape[0]-block_size+1, step):
		for j in range(0,histograms.shape[1]-block_size+1, step):
			normalized.extend((norm_f(histograms[i:i+block_size, 
				j:j+block_size].flatten())))
	return np.asarray(normalized)

def obtainDataFeatures(deriv_kernel, pos_path, neg_path, norm_f = norm_2):
	kx = deriv_kernel[0]
	ky = deriv_kernel[1]

	pos_dir = os.listdir(pos_path)
	neg_dir = os.listdir(neg_path)

	descriptor_length = (15*7)*(4*9)
	features = np.zeros((len(pos_dir) + len(neg_dir), descriptor_length))
	labels = np.repeat(np.asarray([1,0]), [len(pos_dir),len(neg_dir)])
	i = 0

	for name in pos_dir:
		img = cv2.imread(os.path.join(pos_path, name))

		magnitudes, angles = computeGradient(img, kx, ky)

		border_s = (img.shape[0]-128) // 2
		histograms = computeCellHistograms(border_s, magnitudes, angles, 8)
		descriptor = normalizeHistograms(histograms,norm_f=norm_f)
		features[i,:] = descriptor
		i+=1

	for name in neg_dir:
		img = cv2.imread(os.path.join(neg_path, name))

		magnitudes, angles = computeGradient(img, kx, ky)

		border_s = (img.shape[0]-128) // 2
		histograms = computeCellHistograms(border_s, magnitudes, angles, 8)
		descriptor = normalizeHistograms(histograms,norm_f=norm_f)
		features[i,:] = descriptor
		i+=1

	return features, labels

# Warning: this function writes data to disk
def extractNegativeWindows(path, dst_path):
	for name in os.listdir(path):
		img = cv2.imread(os.path.join(path, name))
		for i in range(10):
			f = random.randint(0,img.shape[0]-128)
			c = random.randint(0,img.shape[1]-64)
			cv2.imwrite(dst_path+"/"+str(i)+"_"+name, img[f:f+128,c:c+64])

def scanForPedestrians(img, classifiers, simple_classifier, 
						deriv_kernel, norm_f, stride = 8):

	dims = (int(img.shape[1]//1.2),int(img.shape[0]//1.2))
	pyramid = [img]
	windows = []

	while dims[0] > 128 and dims[1] > 64:
		blurred = cv2.GaussianBlur(pyramid[-1],ksize=(5,5), sigmaX = 0.6)
		pyramid.append(cv2.resize(src=blurred,dsize=dims))
		dims = (int(dims[0]//1.2),int(dims[1]//1.2))

	scale = 1
	
	for level in pyramid:
		# Margin in each side: top, bottom, left, right
		mt = int((level.shape[0] % 8) // 2)
		mb = int((level.shape[0] % 8) - mt)
		ml = int((level.shape[1] % 8) // 2)
		mr = int((level.shape[1] % 8) - ml)

		magnitudes, angles = computeGradient(level[mt:level.shape[0]-mb,
									ml:level.shape[1]-mr],*deriv_kernel)
		histograms = computeCellHistograms(0,magnitudes,angles)

		for i in range(0,histograms.shape[0]-15):
			for j in range(0,histograms.shape[1]-7):
				window_features = normalizeHistograms(histograms[i:i+16,j:j+8],
					norm_f=norm_f)
				prob = simple_classifier.predict_proba([window_features])
				if prob[0,1] > 0.8:
					confidence = heavy_classifier([window_features], 
													classifiers)
					if confidence > 0.5:
						# coordinates x,y
						windows.append(np.array([j*8*scale,i*8*scale,
						(j*8+64)*scale, (i*8+128)*scale, confidence])) 

		scale *= 1.2

	return non_maximum_suppression(np.asarray(windows), 0.3)

def non_maximum_suppression(windows, overlap_threshold):
	if not len(windows):
		return np.array([])

	# windows[:,0], windows[:,1] contain x,y of top left corner
	# windows[:,2], windows[:,3] contain x,y of bottom right corner
	I = np.argsort(windows[:,4])[::-1]
	area = (windows[:,2]-windows[:,0]+1) * (windows[:,3]-windows[:,1]+1)
	chosen = []

	while len(I):
		i = I[0]
		# Dims of intersections between window i and the rest
		width = np.maximum(0.0, np.minimum(windows[i,2], windows[I,2])-
			np.maximum(windows[i,0], windows[I,0])+1)
		height = np.maximum(0.0, np.minimum(windows[i,3], windows[I,3])-
			np.maximum(windows[i,1], windows[I,1])+1)

		overlap = (width*height).astype(np.float32)/area[I]
		mask = overlap<overlap_threshold

		I = I[mask]
		if mask.shape[0]-np.sum(mask) > 1 :
			chosen.append(i)
	return windows[chosen]

def heavy_classifier(window, classifier_list):
	votes = 0
	for classifier in classifier_list:
		votes += classifier.predict(window)
	return votes/len(classifier_list)

def train_classifier_set(features_yes, features_no, n):
	classifiers = []
	c = svm.LinearSVC(C = 0.01)
	labels = np.repeat(np.asarray([1,0]), [len(features_yes),len(features_yes)])

	for i in range(n):
		samples_no = random.sample(range(len(features_no)), len(features_yes))
		c.fit(np.concatenate((features_yes, features_no[samples_no])),labels)
		classifiers.append(c)
	return classifiers

def scanForPedestriansSimple(img,classifier,deriv_kernel,norm_f,stride = 8):

	dims = (int(img.shape[1]//1.2),int(img.shape[0]//1.2))
	pyramid = [img]
	windows = []

	while dims[0] > 128 and dims[1] > 64:
		blurred = cv2.GaussianBlur(pyramid[-1],ksize=(5,5), sigmaX = 0.6)
		pyramid.append(cv2.resize(src=blurred,dsize=dims))
		dims = (int(dims[0]//1.2),int(dims[1]//1.2))

	scale = 1

	for level in pyramid:
		# Margin in each side: top, bottom, left, right
		mt = int((level.shape[0] % 8) // 2)
		mb = int((level.shape[0] % 8) - mt)
		ml = int((level.shape[1] % 8) // 2)
		mr = int((level.shape[1] % 8) - ml)

		magnitudes, angles = computeGradient(level[mt:level.shape[0]-mb,
									ml:level.shape[1]-mr],*deriv_kernel)
		histograms = computeCellHistograms(0,magnitudes,angles)

		for i in range(0,histograms.shape[0]-15):
			for j in range(0,histograms.shape[1]-7):
				window_features = normalizeHistograms(histograms[i:i+16,j:j+8],
					norm_f=norm_f)

				if classifier.predict([window_features])[0]:
					# coordinates x,y
					windows.append(np.array([j*8*scale,i*8*scale,
					(j*8+64)*scale, (i*8+128)*scale, 1])) 

		scale *= 1.2

	return windows

def scanForPedestriansNMS(img,classifier,deriv_kernel,norm_f,stride = 8):

	dims = (int(img.shape[1]//1.2),int(img.shape[0]//1.2))
	pyramid = [img]
	windows = []

	while dims[0] > 128 and dims[1] > 64:
		blurred = cv2.GaussianBlur(pyramid[-1],ksize=(5,5), sigmaX = 0.6)
		pyramid.append(cv2.resize(src=blurred,dsize=dims))
		dims = (int(dims[0]//1.2),int(dims[1]//1.2))

	scale = 1

	for level in pyramid:
		# Margin in each side: top, bottom, left, right
		mt = int((level.shape[0] % 8) // 2)
		mb = int((level.shape[0] % 8) - mt)
		ml = int((level.shape[1] % 8) // 2)
		mr = int((level.shape[1] % 8) - ml)

		magnitudes, angles = computeGradient(level[mt:level.shape[0]-mb,
									ml:level.shape[1]-mr],*deriv_kernel)
		histograms = computeCellHistograms(0,magnitudes,angles)

		for i in range(0,histograms.shape[0]-15):
			for j in range(0,histograms.shape[1]-7):
				window_features = normalizeHistograms(histograms[i:i+16,j:j+8],
														norm_f=norm_f)
				confidence = classifier.decision_function([window_features])[0]
				if confidence > 0:
					# coordinates x,y
					windows.append(np.array([j*8*scale,i*8*scale,
					(j*8+64)*scale, (i*8+128)*scale, confidence])) 

		scale *= 1.2

	return non_maximum_suppression(np.asarray(windows), 0.3)

def scanForPedestriansNMS_votes(img,classifiers,deriv_kernel,norm_f,stride = 8):

	dims = (int(img.shape[1]//1.2),int(img.shape[0]//1.2))
	pyramid = [img]
	windows = []

	while dims[0] > 128 and dims[1] > 64:
		blurred = cv2.GaussianBlur(pyramid[-1],ksize=(5,5), sigmaX = 0.6)
		pyramid.append(cv2.resize(src=blurred,dsize=dims))
		dims = (int(dims[0]//1.2),int(dims[1]//1.2))

	scale = 1

	for level in pyramid:
		# Margin in each side: top, bottom, left, right
		mt = int((level.shape[0] % 8) // 2)
		mb = int((level.shape[0] % 8) - mt)
		ml = int((level.shape[1] % 8) // 2)
		mr = int((level.shape[1] % 8) - ml)

		magnitudes, angles = computeGradient(level[mt:level.shape[0]-mb,
			ml:level.shape[1]-mr],*deriv_kernel)
		histograms = computeCellHistograms(0,magnitudes,angles)

		for i in range(0,histograms.shape[0]-15):
			for j in range(0,histograms.shape[1]-7):
				window_features = normalizeHistograms(histograms[i:i+16,j:j+8],
														norm_f=norm_f)
				confidence = heavy_classifier([window_features], classifiers)
				if confidence > 0.5:
					# coordinates x,y
					windows.append(np.array([j*8*scale,i*8*scale,
					(j*8+64)*scale, (i*8+128)*scale, confidence]))

		scale *= 1.2

	return non_maximum_suppression(np.asarray(windows), 0.3)

def scanForPedestriansNMS_filter(img, classifier, simple_classifier,
									deriv_kernel, norm_f, stride = 8):

	dims = (int(img.shape[1]//1.2),int(img.shape[0]//1.2))
	pyramid = [img]
	windows = []

	while dims[0] > 128 and dims[1] > 64:
		blurred = cv2.GaussianBlur(pyramid[-1],ksize=(5,5), sigmaX = 0.6)
		pyramid.append(cv2.resize(src=blurred,dsize=dims))
		dims = (int(dims[0]//1.2),int(dims[1]//1.2))

	scale = 1

	for level in pyramid:
		# Margin in each side: top, bottom, left, right
		mt = int((level.shape[0] % 8) // 2)
		mb = int((level.shape[0] % 8) - mt)
		ml = int((level.shape[1] % 8) // 2)
		mr = int((level.shape[1] % 8) - ml)

		magnitudes, angles = computeGradient(level[mt:level.shape[0]-mb,
									ml:level.shape[1]-mr],*deriv_kernel)
		histograms = computeCellHistograms(0,magnitudes,angles)

		for i in range(0,histograms.shape[0]-15):
			for j in range(0,histograms.shape[1]-7):
				window_features = normalizeHistograms(histograms[i:i+16,j:j+8],
														norm_f=norm_f)
				prob = simple_classifier.predict_proba([window_features])

				if prob[0,1] > 0.8:
					conf = classifier.decision_function([window_features])[0]
					if conf > 0:
						# coordinates x,y
						windows.append(np.array([j*8*scale,i*8*scale,
						(j*8+64)*scale, (i*8+128)*scale, conf])) 

		scale *= 1.2

	return non_maximum_suppression(np.asarray(windows), 0.3)

def showClassifierStats(features_train, labels_train, features_test, 
		labels_test, classifier, name_classifier, name_norm):
	classifier.fit(features_l1,labels)
	t0 = time.perf_counter()
	prediction = clasificador_svm_a.predict(features_l1_test)
	t1 = time.perf_counter()
	correct_answers = np.sum(np.equal(prediction, labels_test))
	print("Test %s %s: %.5f" % (name_norm, name_classifier, correct_answers/len(prediction)))
	print("Test time: %.5f sec, %.5f each" % (t1-t0, (t1-t0)/labels_test.shape[0]))
	print(metrics.confusion_matrix(labels_test, prediction))

	return classifier

def showStatAndResultFromScan(img, scanning, classifier, kernel, norm, 
						img_id, name = "", second_classifier = None):
	if second_classifier == None:
		t0 = time.perf_counter()
		result = scanning(img, classifier,kernel,norm)
		t1 = time.perf_counter()
	else:
		t0 = time.perf_counter()
		result = scanning(img, classifier, second_classifier ,kernel,norm)
		t1 = time.perf_counter()
	
	canvas = np.copy(img)
	for win in result:
		canvas = cv2.rectangle(canvas,(int(win[0]),int(win[1])),(int(win[2]),int(win[3])),(0,255,0))

	
	print("%s img%d: %.5f sec" % (name, img_id, t1-t0))
	h.showMatrix([canvas], [name], grey= False, col= 1, row=1)

def main():

	pos_train_path = "../INRIAPerson/96X160H96/Train/pos"
	neg_train_path = "../INRIAPerson/96X160H96/Train/neg"
	
	pos_test_path = "../INRIAPerson/70X134H96/Test/pos"
	neg_test_path = "../INRIAPerson/70X134H96/Test/neg"

	kx = np.asarray([[-1],[0],[1]])
	ky = np.asarray([[0],[1],[0]])

	# Commented since both write on disk
	#	extractNegativeWindows("../INRIAPerson/Train/neg", neg_train_path)
	#	extractNegativeWindows("../INRIAPerson/Test/neg", "../INRIAPerson/70X134H96/Test/neg")

	features_l1, labels = obtainDataFeatures((kx,ky), pos_train_path, neg_train_path, norm_1)
	features_l1_test, labels_test = obtainDataFeatures((kx,ky), pos_test_path, neg_test_path, norm_1)
	
	features_l2, _ = obtainDataFeatures((kx,ky), pos_train_path, neg_train_path, norm_2)
	features_l2_test, _ = obtainDataFeatures((kx,ky), pos_test_path, neg_test_path, norm_2)

	features_l2_hys, _ = obtainDataFeatures((kx,ky), pos_train_path, neg_train_path, norm_2_hys)
	features_l2_hys_test, _ = obtainDataFeatures((kx,ky), pos_test_path, neg_test_path, norm_2_hys)

	clasificador_svm = [svm.LinearSVC( C = 0.01, class_weight = 'balanced')]
	clasificador_svm.append(svm.LinearSVC( class_weight = 'balanced'))
	clasificador_svm.append(svm.LinearSVC( C = 0.01))
	clasificador_svm.append(svm.LinearSVC())
	nombres = ["A", "B", "C", "D"]
	
	clasificador_svm_l1 = []
	for i in range(len(clasificador_svm)):
		clasificador_svm_l1.append(showClassifierStats(features_l1, labels,
			features_l1_test, labels_test, clasificador_svm[i], nombres[i], "L1"))
	
	clasificador_svm_l2 = []
	for i in range(len(clasificador_svm)):
		clasificador_svm_l2.append(showClassifierStats(features_l2, labels,
			features_l2_test, labels_test, clasificador_svm[i], nombres[i], "L2"))
	
	clasificador_svm_l2_hys = []
	for i in range(len(clasificador_svm)):
		clasificador_svm_l2_hys.append(showClassifierStats(features_l2_hys, labels,
			features_l2_hys_test, labels_test, clasificador_svm[i], nombres[i], "L2_Hys"))

	del clasificador_svm_l1
	del clasificador_svm_l2_hys
	del features_l1
	del features_l1_test
	del features_l2_hys
	del features_l2_hys_test

	# Test with several classifiers

	clasificador = clasificador_svm_l2[2] # C = 0.01

	# Logistic Regression
	simple_classifier = linear_model.LogisticRegression()
	simple_classifier.fit(features_l2, labels)
	simple_predict = simple_classifier.predict_proba(features_l2_test)[:,1]
	simple_predict = (simple_predict*10)//8
	correct_answers = np.sum(np.equal(simple_predict, labels_test))
	print("Test LogReg  : %.4f" % (correct_answers/len(simple_predict)))
	print(metrics.confusion_matrix(labels_test, simple_predict))

	# 11 SVM, simple majority
	classifiers = train_classifier_set(features_l2[:num_yes], features_l2[num_yes:],11)
	num_yes = np.sum(labels)

	heavy_predict = np.rint(heavy_classifier(features_l2_test, classifiers))
	correct_answers = np.sum(np.equal(heavy_predict, labels_test))
	print("Test SVM vote: %.4f" % (correct_answers/len(heavy_predict)))
	print(metrics.confusion_matrix(labels_test, heavy_predict))
	
	# Test images
	test_image_path_1 = "../INRIAPerson/Test/pos/crop001573.png"
	test_image_path_2 = "../INRIAPerson/Test/pos/crop001684.png"
	test_image_path_3 = "../INRIAPerson/Test/pos/crop001670.png"
	test_image_path_4 = "../INRIAPerson/Test/neg/no_person__no_bike_123.png"
		
	test_image = [cv2.imread(test_image_path_1)]
	test_image.append(cv2.imread(test_image_path_2))
	test_image.append(cv2.imread(test_image_path_3))
	test_image.append(cv2.imread(test_image_path_4))
	i =0
	for img in test_image:

		showStatAndResultFromScan(img, scanForPedestriansSimple, clasificador,
			(kx,ky), norm_2, i, "SVM sin NMS")

		showStatAndResultFromScan(img, scanForPedestriansNMS, clasificador,
			(kx,ky), norm_2, i, "SVM con NMS")

		showStatAndResultFromScan(img, scanForPedestriansNMS_filter, clasificador,
			(kx,ky), norm_2, i, "RegLog y SVM con NMS", simple_classifier)

		showStatAndResultFromScan(img, scanForPedestriansNMS_votes, classifiers,
			(kx,ky), norm_2, i, "votación SVM con NMS")
		
		i+=1

if __name__ == "__main__":
	main()
