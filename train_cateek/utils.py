# coding=utf-8

import copy
import os
import random
import pickle
import time
import sys
import collections
import numpy as np
import pandas as pd

import torch
import torch.optim
import sklearn.metrics
import albumentations as A
import torch.nn.functional as F
import pretrainedmodels

from torch import nn
from torchsummary import summary
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import lr_scheduler
from torch.nn.parameter import Parameter
from albumentations.pytorch import ToTensor
from efficientnet_pytorch import EfficientNet

import dataloader
from opt import *


def parse_args(args):
	n_args = {}
	for k, v in args.items():
		if v == '' or v == 'True':
			n_args[k.replace('--', '')] = bool(v)
		else:
			n_args[k.replace('--', '')] = v
			try:
				n_args[k.replace('--', '')] = int(v)
			except ValueError:
				pass
	return n_args


def load_weights(weights):
	state_dict = torch.load(weights)
	new_state_dict = collections.OrderedDict()
	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v
	return new_state_dict


def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	

def save_model(model, optim, detail, fold, dirname):
	if not os.path.isdir(dirname):
		os.mkdir(dirname)
	path = os.path.join(dirname, 'fold%d_ep%d.pt' % (fold, detail['epoch']))
	torch.save({
		'model': model.state_dict(),
		'optim': optim.state_dict(),
		'detail': detail,
	}, path)
	print('saved model to %s' % path)


def load_model(path, model, gpu, optim=None, dif_loss=False):
	state = torch.load(str(path), map_location=lambda storage, location: storage)
	if dif_loss:
		model.load_state_dict(state['model'])
		model.module.head[-1] = nn.Linear(model.module.head[-1].in_features, 6, bias=False)
	else:
		model.load_state_dict(state['model'])
	if optim:
		print('loading optim too')
		optim.load_state_dict(state['optim'])
	else:
		print('not loading optim')
	
	if gpu is not None:
		model.cuda(gpu)
		detail = state['detail']
		if gpu == 0:
			print('loaded model from %s' % path)
	else:
		model.cuda()
		detail = state['detail']
		print('loaded model from %s' % path)
	
	return detail
	

def get_optim(name, params, lr):
	if name == 'RAdam':
		optim = RAdam(params, lr)
	elif name == 'AdamW':
		optim = AdamW(params, lr, warmup=3)
	return optim


def get_lr(optim):
	if optim:
		return optim.param_groups[0]['lr']
	else:
		return 0
	

def drop_bad(path, bad):
	df = pd.read_csv(path)
	bad = pd.read_csv(bad)
	df = df[df.image_id != '3790f55cad63053e956fb73027179707']
	bad = df['image_id'].isin(bad['image_id'])
	df.drop(df[bad].index, inplace=True)
	return df


def embed_output(output):
	thrs = [0.5, 1.5, 2.5, 3.5, 4.5]
	output[output < thrs[0]] = 0
	output[(output >= thrs[0]) & (output < thrs[1])] = 1
	output[(output >= thrs[1]) & (output < thrs[2])] = 2
	output[(output >= thrs[2]) & (output < thrs[3])] = 3
	output[(output >= thrs[3]) & (output < thrs[4])] = 4
	output[output >= thrs[4]] = 5
	return output

