# coding=utf-8

import os
import re
import sys
import cv2
import torch
import random
import pyvips
import numpy as np
import albumentations
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
from skimage.util.shape import view_as_windows

import utils


def pad_if_needed(img, tile):
	if img.shape[0] % tile != 0 or img.shape[1] % tile != 0:
		pad1 = int(np.ceil(img.shape[0] / tile))
		pad1 = tile * pad1 - img.shape[0]
		pad2 = int(np.ceil(img.shape[1] / tile))
		pad2 = tile * pad2 - img.shape[1]
		img_pad = np.pad(img, ((pad1, 0), (pad2, 0), (0, 0)), constant_values=255)
		return img_pad
	else:
		return img


def crop(img, tol=254):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	mask = gray < tol
	img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
	img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
	img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
	img = np.stack([img1, img2, img3], axis=-1)
	
	_, img = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO)
	return img


def transforms_train(img):
	transforms = albumentations.Compose([
		albumentations.OneOf([
			albumentations.RGBShift(p=1),
			albumentations.RandomGamma(p=1),
		], p=0.5),
		albumentations.RandomBrightnessContrast(p=0.7),
		albumentations.OneOf([
			albumentations.RandomRotate90(p=1),
			albumentations.Flip(p=1),
			albumentations.Rotate(limit=10, border_mode=0, value=(255, 255, 255), p=1),
			albumentations.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.1,
											rotate_limit=10, border_mode=0, value=(255, 255, 255), p=1),
		], p=0.25),
		albumentations.OneOf([
			albumentations.Cutout(num_holes=50, max_h_size=10, max_w_size=10, fill_value=0, p=1),
			albumentations.Cutout(num_holes=70, max_h_size=7, max_w_size=7, fill_value=0, p=1),
			albumentations.Cutout(num_holes=100, max_h_size=5, max_w_size=5, fill_value=0, p=1),
		], p=0.2),
	])
	return transforms(image=img)['image']


def transforms_tile(img):
	transforms = albumentations.Compose([
		albumentations.Transpose(),
		albumentations.VerticalFlip(),
		albumentations.HorizontalFlip(),
	])
	return transforms(image=img)['image']


def transforms_val(img):
	transforms = albumentations.Compose([
		albumentations.NoOp()
	])
	return transforms(image=img)['image']


def crop_val(img, factor=0.95):
	transforms = albumentations.Compose([
		albumentations.CenterCrop(int(img.shape[0] * factor), int(img.shape[1]*factor)),
	])
	return transforms(image=img)['image']


def preprocess_init(img, factor=0.95):
	transforms = albumentations.Compose([
		albumentations.RandomCrop(int(img.shape[0] * factor), int(img.shape[1] * factor)),
		albumentations.Rotate(limit=5, border_mode=0, value=(255,255,255), p=1),
	])
	return transforms(image=img)['image']


def noop(img):
	transforms = albumentations.Compose([
		albumentations.NoOp()
	])
	return transforms(image=img)['image']


def windows(img_fn, tiles, transforms, mode, tile_size=256, overlap=1, crop_down=-1, threshold=240):
	transforms, preprocess = transforms
	overlap = float(overlap)# + r
	n_tiles = []
	res = np.array(Image.open(img_fn))
	
	res = preprocess(res)
	res = pad_if_needed(res, tile_size)
	
	windows = view_as_windows(res, (tile_size, tile_size, 1), (int(tile_size*overlap), int(tile_size*overlap), 1))[..., 0].transpose(0, 1, 3, 4, 2)
	windows_orig = windows.reshape(windows.shape[0] * windows.shape[1], windows.shape[2], windows.shape[3], windows.shape[4])
	mean = np.mean(windows_orig, axis=(1, 2, 3))
	windows = np.delete(windows_orig, np.where(mean >= threshold), axis=0)

	if windows.shape[0] == 0:
		windows = windows_orig
	
	if windows.shape[0] < tiles:
		n_tiles.append(windows.shape[0])
		idxs = np.argsort(windows.reshape(windows.shape[0], -1).sum(-1))
		windows = [transforms(windows[i]) for i in range(windows.shape[0])]
		windows = np.stack(windows)
		
		windows_pad = np.repeat(255*np.ones_like(np.expand_dims(windows[0], 0)), tiles - windows.shape[0], axis=0)
		windows = np.concatenate((windows, windows_pad), axis=0)
	else:
		if mode == 'train':
			idxs = np.argsort(windows.reshape(windows.shape[0], -1).sum(-1))[:tiles // 3]  # [::-1]
			idxs_r = np.argsort(windows.reshape(windows.shape[0], -1).sum(-1))[tiles//3:]
			idxs_n = np.concatenate((idxs, idxs_r))[:tiles]
		else:
			idxs_n = np.argsort(windows.reshape(windows.shape[0], -1).sum(-1))[:tiles]
		
		n_tiles.append(windows.shape[0])
		windows = windows[idxs_n]
		windows = [transforms(windows[i]) for i in range(windows.shape[0])]
		windows = np.stack(windows)
		
	return windows, n_tiles


class PANDA_dataset(torch.utils.data.Dataset):
	def __init__(self, windows, mode, df, path, fold, tiles, tile_size, overlap):
		self.df = df
		self.path = path
		self.fold = fold
		self.mode = mode
		self.windows = windows
		self.tiles = tiles
		self.tile_size = tile_size
		self.overlap = overlap
		
		if self.mode == 'valid':
			self.df = self.df.loc[self.df.split == self.fold]
			print('read dataset (%d records)' % len(self.df))
		else:
			self.df = self.df.loc[self.df.split != self.fold]
			print('read dataset (%d records)' % len(self.df))
	
	def __len__(self):
		return len(self.df)
	
	def __getitem__(self, idx):
		row = self.df.iloc[idx]
		img_path = '%s%s.png' % (self.path, row.image_id)
		if self.mode == 'train':
			imr_array, _ = self.windows(img_path, self.tiles, [transforms_train, preprocess_init], 'train', self.tile_size, self.overlap)
		else:
			imr_array, _ = self.windows(img_path, self.tiles, [transforms_val, noop], 'val', self.tile_size, self.overlap)
		
		return torch.tensor(imr_array.transpose(0, 3, 1, 2)).float()/255,\
			   torch.tensor(row.isup_grade, dtype=torch.float)


def get_dataloader(mode, df, path, fold, batch_size, tiles, tile_size, overlap, rank, num_workers=16):
	dataset = PANDA_dataset(windows, mode, df, path, fold, tiles, tile_size, overlap)
	if rank is not None:
		shuffle = False
		num_workers = 2
		sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=4, rank=rank)
		
	else:
		shuffle = (mode=='train')
		sampler = None
		
	loader = torch.utils.data.DataLoader(dataset,
										 batch_size=batch_size,
										 num_workers=num_workers,
										 shuffle=shuffle,
										 sampler=sampler,
										 pin_memory=False,
										 drop_last=(mode=='train')
										 )
	return loader

