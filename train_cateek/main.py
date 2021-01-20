# coding=utf-8

"""
usage: main.py [options]

options:
    --batch_size=bs          input batch size for training [default: 4]
    --fold=f                 Fold to train on [default: 0]
    --epochs=e               number of epochs to train [default: 60]
    --iteration=i            Experiment number [default: 6]
    --lr=lr                  learning rate [default: 4e-4]
    --load_optim=lo          If load optim too, bool [default: ]
    --mode=m                 Mode [default: train]
    --model=md               Model name [default: Resnext50_ws]
    --n_grad_acc=nga         Gradient accumulations steps [default: 1]
    --optim=o                Optimizer [default: RAdam]
    --overlap=ol             Tile window overlap [default: 1]
    --seed=s                 random seed [default: 10]
    --resume=r               If resume from checkpoint, bool [default: True]
    --tiles=t                Number of tiles [default: 49]
    --tile_size=ts           Tile size [default: 224]
    -h, --help               Show this help message and exit
"""

import sys
import os
import time
import json

import numpy as np
import pandas as pd
from docopt import docopt
from sklearn.metrics import cohen_kappa_score

import apex
import torch
import torch.nn.functional as F
from torch import nn


import utils as utils
from dataloader import *
from models import Resnext50_ws


with open('SETTINGS.json', 'r') as fp:
	settings = json.load(fp)
	path = settings['CLEAN_DATA_DIR']
	workdir = settings['WORKDIR']
	resume_from = settings['MODEL_CHECKPOINT_DIR']
	main_df = settings['TRAIN_DATA_CLEAN_PATH']
	bad_df = settings['BAD_SLIDES']
	
	
df = utils.drop_bad(main_df, bad_df)


def main():
	args = utils.parse_args(docopt(__doc__))
	print(args)
	train(args)


def train(args):
	model = Resnext50_ws()
	model.cuda()
	optim = utils.get_optim(args['optim'], model.parameters(), lr=float(args['lr']))
	
	criterion = nn.SmoothL1Loss()
	
	best = {
		'loss': float('inf'),
		'score': 0.0,
		'epoch': -1,
	}
	
	loader_train = get_dataloader(args['mode'],
								  df,
								  path,
								  args['fold'],
								  args['batch_size'],
								  args['tiles'],
								  args['tile_size'],
								  args['overlap'],
								  rank=None
								  )
	
	loader_valid = get_dataloader('valid',
								  df,
								  path,
								  args['fold'],
								  args['batch_size'],
								  args['tiles'],
								  args['tile_size'],
								  args['overlap'],
								  rank=None
								  )
	
	print('train data: loaded %d records' % len(loader_train.dataset))
	print('valid data: loaded %d records' % len(loader_valid.dataset))
	
	
	apex.amp.initialize(model, optim,
						opt_level='O1',
						verbosity=1,
						min_loss_scale=1,
						#max_loss_scale=1024*16
						)
	
	model = nn.DataParallel(model)
	
	if args['resume']:
		if args['load_optim']:
			detail = utils.load_model(resume_from, model, gpu=None, optim=optim)
		else:
			detail = utils.load_model(resume_from, model, gpu=None, optim=None)
			
		best.update({
			'loss': detail['loss'],
			'score': detail['score'],
			#'epoch': detail['epoch'],
		})
		
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max', factor=0.5, patience=5, verbose=True)
	
	for epoch in range(best['epoch'] + 1, args['epochs']):
		
		print(f'\n----- epoch {epoch} -----')

		utils.set_seed(epoch)
		if args['mode'] == 'train':
			tr = run_nn('train', model, loader_train, args['n_grad_acc'], criterion=criterion, optim=optim)
			
		with torch.no_grad():
			val = run_nn('valid', model, loader_valid, args['n_grad_acc'], criterion=criterion)
		
		detail = {
			'score': val['score'],
			'loss': val['loss'],
			'targets': val['targets'],
			'outputs_raw': val['outputs_raw'],
			'epoch': epoch,
		}
		
		if detail['score'] >= best['score']:
			best.update(detail)
		
		print('[current_val] ep:%d loss:%.4f score:%.4f' % (detail['epoch'], detail['loss'], detail['score']))
		print('[best] ep:%d loss:%.4f score:%.4f' % (best['epoch'], best['loss'], best['score']))
		utils.save_model(model, optim, detail, args['fold'], workdir + args['model'] + '_' + str(args['iteration']))
		
		scheduler.step(val['score']) #reducelronplateau
	

def run_nn(mode, model, loader, n_grad_acc, criterion=None, optim=None):
	if mode == 'train':
		model.train()
	else:
		model.eval()

	t1 = time.time()
	losses = []
	ids_all = []
	targets_all = []
	outputs_all = []
	outputs_all_raw = []
	
	for i, (inputs, targets) in enumerate(loader):
		outputs = model(inputs.cuda())
		labels = targets.cuda(non_blocking=True)
		loss = criterion(outputs.view(-1), labels)
		
		with torch.no_grad():
			targets_all.extend(labels.cpu().numpy())
			outputs_all.extend(utils.embed_output(outputs.cpu().numpy()))
			outputs_all_raw.extend(outputs.cpu().numpy())
			losses.append(loss.item())
		
		if mode == 'train':
			with apex.amp.scale_loss(loss, optim) as scaled_loss:
				scaled_loss.backward()
			if (i + 1) % n_grad_acc == 0:
				optim.step()
				optim.zero_grad()
		
		elapsed = int(time.time() - t1)
		eta = int(elapsed / (i + 1) * (len(loader) - (i + 1)))
		progress = f'\r[{mode}] {i+1}/{len(loader)} {elapsed}(s) eta:{eta}(s) loss:{(np.sum(losses)/(i+1)):.6f} lr:{utils.get_lr(optim):.2e}'
		print(progress, end=' ')
		sys.stdout.flush()
	
	result = {
		'ids': ids_all,
		'targets': np.array(targets_all),
		'outputs': np.array(outputs_all),
		'outputs_raw': np.array(outputs_all_raw),
		'loss': np.sum(losses) / (i + 1),
	}
	
	result['score'] = cohen_kappa_score(result['targets'], result['outputs'], weights='quadratic')
	print(progress + ' kappa_score:%.4f' % (result['score']))
	
	return result


if __name__ == '__main__':
	
	torch.backends.cudnn.benchmark = True
	torch.backends.cudnn.deterministic = True
	main()
