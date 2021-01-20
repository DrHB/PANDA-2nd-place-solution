# coding=utf-8
import os
import json
import collections
import pandas as pd
import numpy as np
import skimage
from skimage.util.shape import view_as_windows

import torch

from dataloader import crop, pad_if_needed
from models import Resnext50_ws, fix_weights
from utils import embed_output


with open('SETTINGS.json', 'r') as fp:
	settings = json.load(fp)
	path = settings['RAW_DATA_TEST']
	resume_from = settings['MODEL_CHECKPOINT_DIR']
	submission = settings['TEST_DATA_CLEAN_PATH']
	submission_dir = settings['SUBMISSION_DIR']
	

def load_weights(weights):
	state_dict = torch.load(weights)['model']
	new_state_dict = collections.OrderedDict()
	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v
	return new_state_dict


def windows(img_fn, tiles=49, tile_size=224, overlap=1, threshold=254):
	res = skimage.io.MultiImage(img_fn)[-2]
	res = crop(res)
	res = pad_if_needed(res, tile_size)
	
	windows = view_as_windows(res, (tile_size, tile_size, 1), (int(tile_size * overlap), int(tile_size * overlap), 1))[
		..., 0].transpose(0, 1, 3, 4, 2)
	windows_orig = windows.reshape(windows.shape[0] * windows.shape[1], windows.shape[2], windows.shape[3],
								   windows.shape[4])
	mean = np.mean(windows_orig, axis=(1, 2, 3))
	windows = np.delete(windows_orig, np.where(mean >= threshold), axis=0)
	
	if windows.shape[0] == 0:
		windows = windows_orig
	idxs = np.argsort(windows.reshape(windows.shape[0], -1).sum(-1))[:tiles]  # [::-1]
	if windows.shape[0] < tiles:
		windows_pad = np.repeat(255 * np.ones_like(np.expand_dims(windows[0], 0)), tiles - windows.shape[0], axis=0)
		windows = np.concatenate((windows, windows_pad), axis=0)
	else:
		windows = windows[idxs]
	
	return windows


class TestDataset(torch.utils.data.Dataset):
	def __init__(self, df, path):
		self.df = pd.read_csv(df)
		self.path = path
	
	def __len__(self):
		return len(self.df)
	
	def __getitem__(self, idx):
		print(self.df)
		row = self.df.iloc[idx]
		img_path = '%s%s.tiff' % (self.path, row.image_id)
		image = windows(img_path)
		return torch.tensor(image.transpose(0, 3, 1, 2)).float() / 255, row.image_id


model = Resnext50_ws().cuda()
model.load_state_dict(load_weights(resume_from))

preds = []
preds_raw = []
names = []


if os.path.exists(path):
	test_dataset = TestDataset(submission, path)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
	
	for i, (images, name) in enumerate(test_loader):
		images = images.cuda()
		
		with torch.no_grad():
			y_preds = model(images)
		
		preds.append(embed_output(y_preds.cpu().numpy()))
		preds_raw.append(y_preds.cpu().numpy())
		names.extend(name)
	
	preds_out = np.concatenate(preds)
	preds_out_raw = np.concatenate(preds_raw).reshape(-1)
	
	result = pd.DataFrame({
		'image_id': names,
		'isup_grade': preds_out_raw
	})
	
	result.to_csv(os.path.join(submission_dir, 'submission.csv'), index=False)
