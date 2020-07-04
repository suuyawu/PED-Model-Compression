"""
prune units
"""
from os import path
import pickle
import numpy as np
import torch
import torch.nn.functional as F

import models
from utils import *
from test_model import test
import config
import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from copy import deepcopy

def run(param, name):
	seed = param['seed']
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	
	dataset, arch, score_name, policy_mode = name.split('_')
	information = np.load('../output/information/{}_{}.npy'.format('_'.join([dataset, arch, score_name]), param['stage']))
	information = np.mean(information, axis=0)
	if param['stage'] == 1:
		model_path = '../output/model/{}_{}.pt'.format('_'.join([dataset, arch]), 0)
		policy_arr = []
		for num_units in param['blocks'][param['arch']]:
			policy_arr.append([1]*num_units)
		if not param['special_TAG'] == '':
			name = '_'.join([
				name,
				param['special_TAG']])
			seed = seed + int(param['special_TAG'])
	else:
		if not param['special_TAG'] == '':
			name = '_'.join([
				name,
				param['special_TAG']])
			seed = seed + int(param['special_TAG'])
		model_path = '../output/model/{}_{}.pt'.format(name, param['stage']-1)
		policy_arr = np.load('../output/policy/{}_{}.npy'.format(name, param['stage']-1), allow_pickle=True)

	train_dataset = fetch_dataset(param['dataset'], split = 'train')
	train_sampler, valid_sampler = validate_dataset(train_dataset)
	data_loader = load_dataset(train_dataset, param['batch_size']['test'], 
		param['shuffle']['test'], param['pin_memory'], param['num_workers'], sampler = valid_sampler)
	
	criterion = nn.CrossEntropyLoss()
	if not param['save_policy']:
		acc = []
		blk = []
		for re in param['hyper']:
			policy0=deepcopy(policy_arr)
			policy, blk_comp = Policy(policy0, information, re, param)
			model = eval('models.{}.{}(dataset = \'{}\', policy = {}, model_path = \'{}\').to(device)'.format(param['model'],param['arch'], param['dataset'], policy, model_path))
			model = nn.DataParallel(model, device_ids=param['GPUs']) if param['parallel'] else model
			acc_comp, _ = test(data_loader, model, criterion)
			print('Experiment with hyper = {} done.'.format(re))
			acc.append(acc_comp)
			blk.append(blk_comp)
		print(acc)
		print(blk)
	else:
		policy, blk_comp = Policy(policy_arr, information, param['hyper'][0], param)
		np.save('../output/policy/{}_{}.npy'.format(name, param['stage']), policy)
		model = eval('models.{}.{}(dataset = \'{}\', policy = {}, model_path = \'{}\').to(device)'.format(param['model'],param['arch'], param['dataset'], policy, model_path))
		model = nn.DataParallel(model, device_ids=param['GPUs']) if param['parallel'] else model
		acc_comp, _ = test(data_loader, model, criterion)
		
		prune_result = {'acc':acc_comp, 'block':blk_comp}
		if not path.exists('../output/result/prune_result.pkl'):
			results ={}
		else:
			with open('../output/result/prune_result.pkl', "rb", buffering=0) as f:
				results = pickle.load(f, encoding='utf-8')
		results['{}_{}'.format(name,param['stage'])] = prune_result
		with open('../output/result/prune_result.pkl', 'wb') as f:
			pickle.dump(results, f)

def main():
	global device
	parser = config.prepare_parser()
	param = vars(parser.parse_args())
	device = torch.device(param['device'])
	name = config.name_from_config(param)
	# if param['dist_url'] == "env://" and args.world_size == -1:
	# 	args.world_size = int(os.environ["WORLD_SIZE"])
	# 	args.distributed = args.world_size > 1 or args.multiprocessing_distributed
	# 	ngpus_per_node = torch.cuda.device_count()
    # if args.multiprocessing_distributed:
    #     # Since we have ngpus_per_node processes per node, the total world_size
    #     # needs to be adjusted accordingly
    #     args.world_size = ngpus_per_node * args.world_size
    #     # Use torch.multiprocessing.spawn to launch distributed processes: the
    #     # main_worker process function
    #     mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))####
    # else:
    #     # Simply call main_worker function
    #     main_worker(args.gpu, ngpus_per_node, args)
	print(param, name)
	run(param, name)

if __name__ == '__main__':
	main()

