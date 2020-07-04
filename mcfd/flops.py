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
import config

def run(param, name):
	if param['stage'] == 0:
		dataset, arch, score_name, policy_mode = name.split('_')
		model_path = '../output/model/{}_{}.pth'.format('_'.join([dataset, arch]), 0)
	else:
		if not param['special_TAG'] == '':
			name = '_'.join([
				name,
				param['special_TAG']])
			seed = seed + int(param['special_TAG'])
		model_path = '../output/model/{}_{}.pth'.format(name, param['stage'])
	model = torch.load(model_path, map_location = device)
	num_flop, num_param = Flops_cnt(model, param, device)
	print(num_flop, num_param)

	if not path.exists('../output/result/test_result.pkl'):
		results = {}
	else:
		with open('../output/result/test_result.pkl', "rb", buffering=0) as f:
			results = pickle.load(f, encoding='utf-8')
	results['{}_{}'.format(name, param['stage'])].update({'num_flop': num_flop})
	with open('../output/result/test_result.pkl', 'wb') as f:
		pickle.dump(results, f)

def main():
	global device
	parser = config.prepare_parser()
	param = vars(parser.parse_args())
	device = torch.device(param['device'])
	name = config.name_from_config(param)
	print(param, name)
	run(param, name)

if __name__ == '__main__':
	main()
