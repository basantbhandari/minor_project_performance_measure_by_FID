# example of calculating the frechet inception distance
import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm

import matplotlib.pyplot as plt
import os
import cv2


 
# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = numpy.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid
 
DATADIR = "/home/basant/Downloads/GAN_WORKSPACE/minor project/paper writing on minor project/performance_parameter/selected"
CATEGORIES = ["real_selected", "fake_selected"]
IMG_HIGHT = 32
IMG_WIDTH = 64
act1 = []
act2 = []

for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    print("Directory number : ", category)
    for img in os.listdir(path):
        if category == "real_selected":
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            print("one image shape = ", img_array.shape)
            print("one image type = ", type(img_array))
            new_array = cv2.resize(img_array, (IMG_HIGHT, IMG_WIDTH))
            print("one image shape = ", new_array.shape)
            print("one image type = ", type(new_array))
            new_array = new_array.flatten()
            print("one image flatten shape = ", new_array.shape)
            print("one image flatten type = ", type(new_array))
            print("one image flatten ndim = ", new_array.ndim)
            act1.append(new_array)
            print("real image appended")
        else:
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            print("one image shape = ", img_array.shape)
            print("one image type = ", type(img_array))
            new_array = cv2.resize(img_array, (IMG_HIGHT, IMG_WIDTH))
            print("one image shape = ", new_array.shape)
            print("one image type = ", type(new_array))
            new_array = new_array.flatten()
            print("one image flatten shape = ", new_array.shape)
            print("one image flatten type = ", type(new_array))
            print("one image flatten ndim = ", new_array.ndim)
            act2.append(new_array)
            print("fake image appended")


act1 = numpy.array(act1)
act2 = numpy.array(act2)

#   # define two collections of activations
# act1 = random(10*2048)
# act1 = act1.reshape((10,2048))
# act2 = random(10*2048)
# act2 = act2.reshape((10,2048))
# fid between act1 and act1
fid = calculate_fid(act1, act1)
print('FID (same): %.3f' % fid)

fid = calculate_fid(act2, act2)
print('FID (same1): %.3f' % fid)


# fid between act1 and act2
fid = calculate_fid(act1, act2)
print('FID (different): %.3f' % fid)

fid = calculate_fid(act2, act1)
print('FID (different) reverse: %.3f' % fid)


