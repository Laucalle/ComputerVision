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

def computeCellHistograms(border_size, magnitudes, angles, cell_size = 8,
 							bins = 9, max_angle = 180):

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

def normalizeHistograms(histograms, norm_f = norm_2, block_size = 2,
														overlapping = 0.5):
	step = int(block_size*(1-overlapping))
	normalized = []
	for i in range(0,histograms.shape[0]-block_size+1, step):
		for j in range(0,histograms.shape[1]-block_size+1, step):
			normalized.extend((norm_f(histograms[i:i+block_size, j:j+block_size].flatten())))
	return np.asarray(normalized)

def obtainDataFeatures(deriv_kernel, pos_path, neg_path, norm_f = norm_2):
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
		descriptor = normalizeHistograms(histograms,norm_f=norm_f)
		features[i,:] = descriptor
		h.printProgressBar(i, features.shape[0])
		i+=1

	for name in neg_directory:
		img = cv2.imread(os.path.join(neg_path, name))

		magnitudes, angles = computeGradient(img, kx, ky)

		border_size = (img.shape[0]-128) // 2
		histograms = computeCellHistograms(border_size, magnitudes, angles, 8)
		descriptor = normalizeHistograms(histograms,norm_f=norm_f)
		features[i,:] = descriptor
		i+=1
		h.printProgressBar(i, features.shape[0])

	return features, labels

# Warning: this function writes data to disk
def extractNegativeWindows(path, dst_path):
	for name in os.listdir(path):
		img = cv2.imread(os.path.join(path, name))
		for i in range(10):
			f = random.randint(0,img.shape[0]-128)
			c = random.randint(0,img.shape[1]-64)
			cv2.imwrite(dst_path+"/"+str(i)+"_"+name, img[f:f+128,c:c+64])

def extractHardExamples(path, deriv_kernel, classifier, stride = 8, norm_f = norm_2):
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
					window_norm_histograms = normalizeHistograms(histograms[i:i+128,j:j+64],norm_f=norm_f)
					if classifier.predict(window_norm_histograms)[0] > 0.5:
						cv2.imwrite("hard_examples/"+str(n_examples)+".png",histograms[i:i+128,j:j+64])

def scanForPedestrians(img,list_classifiers, simple_classiffier,deriv_kernel,norm_f,draw_window=False, stride = 8):

	dims = (int(img.shape[1]//1.2),int(img.shape[0]//1.2))
	pyramid = [img]
	windows = []

	while dims[0] > 128 and dims[1] > 64:
		blurred = cv2.GaussianBlur(pyramid[-1],ksize=(5,5), sigmaX = 0.6)
		pyramid.append(cv2.resize(src=blurred,dsize=dims))
		dims = (int(dims[0]//1.2),int(dims[1]//1.2))

	scale = 1
	if draw_window:
		minicanvas_list = []
		canvas = np.copy(img)
	for level in pyramid:
		margin_r = int((level.shape[0] % 8) // 2)
		margin_er = int((level.shape[0] % 8) - margin_r)
		margin_c = int((level.shape[1] % 8) // 2)
		margin_ec = int((level.shape[1] % 8) - margin_c)

		magnitudes, angles = computeGradient(level[margin_r:level.shape[0]-margin_er,margin_c:level.shape[1]-margin_ec],*deriv_kernel)

		histograms = computeCellHistograms(0,magnitudes,angles)
		if draw_window:
			minicanvas = np.copy(level)
			minicanvas_list.append(minicanvas)
		for i in range(0,histograms.shape[0]-15):
			for j in range(0,histograms.shape[1]-7):
				window_norm_histograms = normalizeHistograms(histograms[i:i+16,j:j+8],norm_f=norm_f)
				prob = simple_classiffier.predict_proba([window_norm_histograms])

				if prob[0,1] > 0.8:
					#confidence = classifier.decision_function([window_norm_histograms])[0]
					confidence = heavy_classifier([window_norm_histograms], list_classifiers)
					if confidence>0.5:
						if draw_window:
							canvas = cv2.rectangle(canvas,(int(j*8*scale),int(i*8*scale)),(int((j*8+64)*scale),int((i*8+128)*scale)),(255,0,0))
						#	minicanvas = cv2.rectangle(minicanvas, (int(j*8), int(i*8)), (int(j*8+64), int(i*8+128)), (255,0,0))
						windows.append(np.array([j*8*scale,i*8*scale,(j*8+64)*scale, (i*8+128)*scale, confidence])) # coordinates x,y

		scale *= 1.2

	print("Starting non maximum suppression ...")
	new_windows = non_maximum_suppression(np.asarray(windows), 0.3)
	new_canvas = np.copy(img)
	for win in new_windows:
		new_canvas = cv2.rectangle(new_canvas,(int(win[0]),int(win[1])),(int(win[2]),int(win[3])),(0,255,0))

	#if draw_window:
	#	h.showMatrix(minicanvas_list, [str(i) for i in range(len(minicanvas_list))], grey = False)
	return canvas, new_canvas

def non_maximum_suppression(windows, overlap_threshold):
	if not len(windows):
		return np.array([])

	x1 = windows[:,0]
	y1 = windows[:,1]
	x2 = windows[:,2]
	y2 = windows[:,3]
	c = windows[:,4]
	I = np.argsort(c)[::-1]

	area = (x2-x1+1) * (y2-y1)+1
	chosen = []

	while len(I):
		i = I[0]
		xx1 = np.maximum(x1[i], x1[I])
		yy1 = np.maximum(y1[i], y1[I])
		xx2 = np.minimum(x2[i], x2[I])
		yy2 = np.minimum(y2[i], y2[I])

		width = np.maximum(0.0, xx2-xx1+1)
		height = np.maximum(0.0, yy2-yy1+1)

		overlap = (width*height).astype(np.float32)/area[I]
		mask = overlap<overlap_threshold

		I = I[mask]
		if mask.shape[0]-np.sum(mask) > 1 :
			chosen.append(i)
	return windows[chosen]

def heavy_classifier(window, classifier_list):
	votes = np.zeros(len(window))
	for classifier in classifier_list:
		votes += classifier.predict(window)
	return votes/len(classifier_list)

def train_classifier_set(features_yes, features_no, n):
	classifiers = []
	c = svm.LinearSVC()
	labels = np.repeat(np.asarray([1,0]), [len(features_yes),len(features_yes)])

	for i in range(n):
		samples_no = random.sample(range(len(features_no)), len(features_yes))
		c.fit(np.concatenate((features_yes, features_no[samples_no])),labels)
		classifiers.append(c)
	return classifiers

def main():

	pos_train_path = "../INRIAPerson/96X160H96/Train/pos"
	neg_train_path = "../INRIAPerson/96X160H96/Train/neg"

	kx = np.asarray([[-1],[0],[1]])
	ky = np.asarray([[0],[1],[0]])

	#test_image_path = "../INRIAPerson/Test/pos/crop001573.png"
	test_image_path = "../INRIAPerson/Test/pos/crop001573.png"
	test_img = cv2.imread(test_image_path)
	#extractNegativeWindows("../INRIAPerson/Train/neg", neg_train_path)
	#extractNegativeWindows("../INRIAPerson/Test/neg", "../INRIAPerson/70X134H96/Test/neg")

	#features, labels = obtainDataFeatures((kx,ky), "../INRIAPerson/70X134H96/Test/pos", "../INRIAPerson/70X134H96/Test/neg", norm_1)
	#features, labels = obtainDataFeatures((kx,ky), "../INRIAPerson/70X134H96/Test/pos", "../INRIAPerson/70X134H96/Test/neg", norm_2_hys)
	#pickle.dump(labels, open("labels_test.pk", "wb"))
	labels_test = pickle.load(open("labels_test.pk", "rb"))
	#features_l1 = pickle.load(open("features_l1_test.pk", "rb"))
	features_l2_hys_test = pickle.load(open("features_l2_hys_test.pk", "rb"))
	#features_l2 = pickle.load(open("features_l2_test.pk", "rb"))


	#t0 = time.perf_counter()
	#features, labels = obtainDataFeatures((kx,ky),pos_train_path,neg_train_path)
	#t1 = time.perf_counter()
	#print("Total L2:    %.5f sec, %.5f each" % (t1-t0, (t1-t0)/labels.shape[0]))
	#pickle.dump(labels, open("labels.pk", "wb"))
	#pickle.dump(features, open("features_l2.pk", "wb"))
	#labels = pickle.load(open("labels.pk", "rb"))
	#features = pickle.load(open("features.pk", "rb"))


	#classifier = svm.LinearSVC()
	#classifier.fit(features, labels)
	#predictions = classifier.predict(features_l2)

	#correct_answers = np.sum(np.equal(predictions,labels))
	#print("Test l2   : %.4f" % (correct_answers/predictions.shape[0]))
	#pickle.dump(classifier, open("classifier_l2.pk","wb"))

	#proc = multiprocessing.Process(target = scanForPedestrians, args = (test_img, classifier,(kx,ky),norm_2, draw_window = True))
	#proc.start()
	#proc.join()

	#predictions = classifier.predict(features)
	#correct_answers = np.sum(np.equal(predictions,labels))
	#print("Test accuracy: %.4f" % (correct_answers/predictions.shape[0]))
	#t0 = time.perf_counter()
	#features, labels = obtainDataFeatures((kx,ky),pos_train_path,neg_train_path, norm_1)
	#t1 = time.perf_counter()
	#print("Total L1:     %.5f sec, %.5f each" % (t1-t0, (t1-t0)/labels.shape[0]))
	#pickle.dump(features, open("features_l1.pk", "wb"))
	#features = pickle.load(open("features_l1.pk", "rb"))

	#classifier = pickle.load(open("classifier_l1.pk", "rb"))

	#classifier.fit(features, labels)
	#predictions = classifier.predict(features_l1)

	#correct_answers = np.sum(np.equal(predictions,labels))
	#print("Test l1   : %.4f" % (correct_answers/predictions.shape[0]))
	#pickle.dump(classifier, open("classifier_l1.pk","wb"))

	#result2 = scanForPedestrians(test_img, classifier,(kx,ky),norm_1, draw_window = True)
	#h.showMatrix([result2], ["L1"], grey = False, col=1, row=1)


	#t0 = time.perf_counter()
	#features, labels = obtainDataFeatures((kx,ky),pos_train_path,neg_train_path, norm_2_hys)
	#t1 = time.perf_counter()
	#print("Total L2-Hys: %.5f sec, %.5f each" % (t1-t0, (t1-t0)/labels.shape[0]))
	#pickle.dump(features, open("features_l2_hys.pk", "wb"))
	#labels = pickle.load(open("labels.pk", "rb"))
	#features = pickle.load(open("features_l2_hys.pk", "rb"))

	#classifier = pickle.load(open("classifier_l2_hys.pk", "rb"))

	#classifier.fit(features, labels)
	#predictions = classifier.predict(features_l2_hys)

	#correct_answers = np.sum(np.equal(predictions,labels))
	#print("Test l2hys: %.4f" % (correct_answers/predictions.shape[0]))
	#pickle.dump(classifier, open("classifier_l2_hys.pk","wb"))

	#result3 = scanForPedestrians(test_img, classifier,(kx,ky),norm_2_hys, draw_window = True)
	#h.showMatrix([result3], ["L2-Hys"], grey = False, col=1, row=1)
	features = pickle.load(open("features_l2_hys.pk", "rb"))
	labels = pickle.load(open("labels.pk", "rb"))

	classifier = pickle.load(open("classifier_l2_hys.pk", "rb"))
	num_yes = np.sum(labels)
	list_classifiers = train_classifier_set(features[:num_yes], features[num_yes:],11)
	#pickle.dump(list_classifiers, open("classifier_list.pk", "wb"))
	#list_classifiers = pickle.load(open("classifier_list.pk","rb"))
	simple_classifier = linear_model.LogisticRegression()
	simple_classifier.fit(features, labels)

	simple_predict = simple_classifier.predict_proba(features_l2_hys_test)[:,1]
	simple_predict = (simple_predict*10)//5
	correct_answers = np.sum(np.equal(simple_predict, labels_test))
	print("Test LogReg  : %.4f" % (correct_answers/len(simple_predict)))
	print(metrics.confusion_matrix(labels_test, simple_predict))

	heavy_predict = np.rint(heavy_classifier(features_l2_hys_test, list_classifiers))
	correct_answers = np.sum(np.equal(heavy_predict, labels_test))
	print("Test SVM vote: %.4f" % (correct_answers/len(heavy_predict)))
	print(metrics.confusion_matrix(labels_test, heavy_predict))

	predictions = classifier.predict(features_l2_hys_test)

	correct_answers = np.sum(np.equal(predictions,labels_test))
	print("Test l2hys   : %.4f" % (correct_answers/len(predictions)))
	print(metrics.confusion_matrix(labels_test, predictions))

	#result1, result2 = scanForPedestrians(test_img, list_classifiers, simple_classifier, (kx,ky),norm_2, draw_window = True)
	#h.showMatrix([result1, result2], ["L2_hys", "L2_hys with NMS"], grey = False, col=2, row=1)

if __name__ == "__main__":
	main()