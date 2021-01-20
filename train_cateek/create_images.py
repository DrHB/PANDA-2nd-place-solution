# coding=utf-8
import os
import cv2
import json
import sys
import zipfile
import pandas as pd
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

from joblib import Parallel, delayed
from PIL import Image
from skimage import io
from skimage import morphology
from tqdm import tqdm

import utils


with open('SETTINGS.json', 'r') as fp:
	settings = json.load(fp)
	raw_images = settings['RAW_DATA_DIR']
	main_path = settings['CLEAN_DATA_DIR']


def otsu_filter(channel, gaussian_blur=True):
	"""Otsu filter."""
	
	if gaussian_blur:
		channel = cv2.GaussianBlur(channel, (5, 5), 0)
	channel = channel.reshape((channel.shape[0], channel.shape[1]))
	
	return cv2.threshold(channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def getSubImage(input_slide, rect):
	"""
	Description
	----------
	Take a cv2 rectagle object and remove its contents from
	a source image.
	Credit: https://stackoverflow.com/a/48553593

	Parameters
	----------
	input_slide: numpy array
			Slide to pull subimage off
	rect: cv2 rect
		cv2 rectagle object with a shape of-
			((center_x,center_y), (hight,width), angle)

	Returns (1)
	-------
	- Numpy array of rectalge data cut from input slide
	"""
	
	width = int(rect[1][0])
	height = int(rect[1][1])
	box = cv2.boxPoints(rect)
	
	src_pts = box.astype("float32")
	dst_pts = np.array([[0, height - 1], [0, 0], [width - 1, 0], [width - 1, height - 1]], dtype="float32")
	
	M = cv2.getPerspectiveTransform(src_pts, dst_pts)
	# input_slide = input_slide.astype(np.float32)
	output_slide = cv2.warpPerspective(input_slide, M, (width, height))
	
	return output_slide


def color_cut(in_slide, color=[255, 255, 255]):
	"""
	Description
	----------
	Take a input image and remove all rows or columns that
	are only made of the input color [R,G,B]. The default color
	to cut from image is white.

	Parameters
	----------
	input_slide: numpy array
		Slide to cut white cols/rows
	color: list
		List of [R,G,B] pixels to cut from the input slide

	Returns (1)
	-------
	- Numpy array of input_slide with white removed
	"""
	# Remove by row
	row_not_blank = [row.all() for row in ~np.all(in_slide == color, axis=1)]
	output_slide = in_slide[row_not_blank, :]
	
	# Remove by col
	col_not_blank = [col.all() for col in ~np.all(output_slide == color, axis=0)]
	output_slide = output_slide[:, col_not_blank]
	return output_slide


def new_detect_and_crop(image_location="", sensitivity: int = 80000, downsample_lvl=-1,
						show_plots="simple", out_lvl=-1, shape=(512, 512)):
	"""
	Description
	----------
	This method performs the pipeline as described in the notebook:
	https://www.kaggle.com/dannellyz/panda-tissue-detect-scaling-bounding-boxes-fast

	Parameters
	----------
	image_location:str
		Location of the slide image to process
	sensitivity:int
		The desired sensitivty of the model to detect tissue. The baseline is set
		at 3000 and should be adjusted down to capture more potential issue and
		adjusted up to be more agressive with trimming the slide.
	downsample_lvl: int
		The level at which to downsample the slide. This can be referenced in
		reverse order to access the lowest resoltuion items first.
		[-1] = lowest resolution
		[0] = highest resolution
	show_plots: str (verbose|simple|none)
		The types of plots to display:
			- verbose - show all steps of process
			- simple - show only last step
			- none - show none of the plots
	out_lvl: int
		The level at which the final slide should sample at. This can be referenced in
		reverse order to access the lowest resoltuion items first.
		[-1] = lowest resolution
		[0] = highest resolution
	shape: touple
		(height, width) of the desired produciton(prod) image

	Returns (4)
	-------
	- Numpy array of final produciton(prod) slide
	- Percent memory reduciton from original slide
	- Time stamps from stages of the pipeline
	- Time stamps from the Tissue Detect pipeline
	"""
	
	wsi_small = io.MultiImage(image_location)[downsample_lvl]
	wsi_big = io.MultiImage(image_location)[out_lvl]
	# wsi_big = io.imread(image_location)
	
	(tissue_contours, tier) = detect_tissue_external(wsi_small, sensitivity)
	
	base_slide_mask = np.zeros(wsi_small.shape[:2])
	
	# Get minimal bounding rectangle for all tissue contours
	if len(tissue_contours) == 0:
		img_id = image_location.split("/")[-1]
		print(f"No Tissue Contours - ID: {img_id}")
		return None, 1.0
	
	# Open Big Slide
	# print(wsi_small.shape, wsi_big.shape, 'shapes')
	# Get small boudning rect and scale
	bounding_rect_small = cv2.minAreaRect(np.concatenate(tissue_contours))
	
	# Scale Rectagle to larger image
	if wsi_big.shape[0] > 32767 or wsi_big.shape[1] > 32767:
		resize_coef = max(wsi_big.shape[0], wsi_big.shape[1]) / 32767
		res_h = int(wsi_big.shape[0] / resize_coef-0.1)
		res_w = int(wsi_big.shape[1] / resize_coef-0.1)
		wsi_big = cv2.resize(wsi_big, (res_w, res_h))
		
	scale = int(wsi_big.shape[0] / wsi_small.shape[0])
	scaled_rect = (
		(bounding_rect_small[0][0] * scale, bounding_rect_small[0][1] * scale),
		(bounding_rect_small[1][0] * scale, bounding_rect_small[1][1] * scale),
		bounding_rect_small[2],
	)
	# Crop bigger image with getSubImage()
	scaled_crop = getSubImage(wsi_big, scaled_rect)
	# Cut out white
	
	#white_cut = color_cut(scaled_crop)
	#print(scaled_crop.shape, 'sc')
	#_, (i1, i2, i3) = plt.subplots(figsize=(10, 10), nrows=3)
	#i1.imshow(crop(wsi_big))
	#i2.imshow(crop(scaled_crop))
	#i3.imshow(scaled_crop)
	#plt.show()
	# Scale
	#scaled_slide = white_cut  #
	img = cv2.resize(wsi_big, (int(wsi_big.shape[1]/2), int(wsi_big.shape[0]/2)), interpolation=cv2.INTER_LANCZOS4)
	
	return crop(img)#scaled_slide


def tissue_cutout(input_slide, tissue_contours):
	"""
	Description
	----------
	Set all parts of the in_slide to black except for those
	within the provided tissue contours
	Credit: https://stackoverflow.com/a/28759496

	Parameters
	----------
	input_slide: numpy array
			Slide to cut non-tissue backgound out
	tissue_contours: numpy array
			These are the identified tissue regions as cv2 contours

	Returns (1)
	-------
	- Numpy array of slide with non-tissue set to black
	"""
	
	# Get intermediate slide
	base_slide_mask = np.zeros(input_slide.shape[:2])
	
	# Create mask where white is what we want, black otherwise
	crop_mask = np.zeros_like(base_slide_mask)
	
	# Draw filled contour in mask
	cv2.drawContours(crop_mask, tissue_contours, -1, 255, -1)
	
	# Extract out the object and place into output image
	tissue_only_slide = np.zeros_like(input_slide)
	tissue_only_slide[crop_mask == 255] = input_slide[crop_mask == 255]
	
	return tissue_only_slide


def detect_tissue_external(input_slide, sensitivity=3000):
	"""
	Description
	----------
	Find RoIs containing tissue in WSI and only return the external most.
	Generate mask locating tissue in an WSI. Inspired by method used by
	Wang et al. [1]_.
	.. [1] Dayong Wang, Aditya Khosla, Rishab Gargeya, Humayun Irshad, Andrew
	H. Beck, "Deep Learning for Identifying Metastatic Breast Cancer",
	arXiv:1606.05718
	Credit: Github-wsipre

	Parameters
	----------
	input_slide: numpy array
		Slide to detect tissue on.
	sensitivity: int
		The desired sensitivty of the model to detect tissue. The baseline is set
		at 3000 and should be adjusted down to capture more potential issue and
		adjusted up to be more agressive with trimming the slide.

	Returns (3)
	-------
	-Tissue binary mask as numpy 2D array,
	-Tiers investigated,
	-Time Stamps from running tissue detection pipeline
	"""
	
	# For timing
	# Convert from RGB to HSV color space
	slide_hsv = cv2.cvtColor(input_slide, cv2.COLOR_BGR2HSV)
	# Compute optimal threshold values in each channel using Otsu algorithm
	_, saturation, _ = np.split(slide_hsv, 3, axis=2)
	
	mask = otsu_filter(saturation, gaussian_blur=True)
	mask = mask != 0
	
	mask = morphology.remove_small_holes(mask, area_threshold=sensitivity)
	mask = morphology.remove_small_objects(mask, min_size=sensitivity)
	mask = mask.astype(np.uint8)
	mask_contours, tier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	
	return mask_contours, tier


def crop(img, tol=254):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	mask = gray < tol
	img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
	img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
	img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
	img = np.stack([img1, img2, img3], axis=-1)
	
	_, img = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO)
	#print(img.shape, 'crop')
	return img


def remove_pen_marks(img):
	# Define elliptic kernel
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
	
	# Reshape red channel into 1-d array, aims to mask most of the pen marks
	img_r = np.reshape(img[:, :, 0], (-1,))
	img_r = img_r[np.where(img_r < 255)[0]]
	
	img_r_mask = (img[:, :, 0] < np.median(img_r) - 90).astype(np.uint8)

	# When computing the pen mark mask, some tissue gets masked as well,
	# thus needing to erode the mask to get rid of it. Then some dilatation is
	# applied to capture the "edges" of the "gradient-like"/non-uniform pen marks
	img_r_mask = cv2.erode(img_r_mask, kernel, iterations=5)
	img_r_mask = cv2.dilate(img_r_mask, kernel, iterations=5)
	
	# Combine the two masks
	img_r_mask = 1 - img_r_mask
	
	# Mask out pen marks from original image
	img = img * img_r_mask[:, :, np.newaxis]
	img = np.where(img==0, 255, img)
	return img


def save_imgs(fn, init_path, res_path):
	img_name = init_path + fn
	print(img_name)
	if fn[:-5] + '.png' not in os.listdir(res_path):
		img = new_detect_and_crop(img_name, show_plots='verbose', out_lvl=-2, sensitivity=1020)
		cv2.imwrite(res_path + fn[:-4] + 'png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


if not os.path.exists(main_path):
	os.makedirs(main_path)
	
Parallel(n_jobs=8)(delayed(save_imgs)(fn, raw_images, main_path) for fn in tqdm(os.listdir(raw_images)))
