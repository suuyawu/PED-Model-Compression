from utils import *
import configs
import itertools
import numpy as np

def generate_policy():
	gpu_ids = ['0', '1', '2', '3']
	script_name = [['Compression_scheme_MI.py']]
	model = [['resnet56']]
	datasets = [['CIFAR10']]
	score_name = [['energy']]
	policy_mode = [['ckmeans']]
	hyper = [[17]]
	stage = [[5]]
	special_TAGs = np.arange(10)
	special_TAGs = special_TAGs.astype(str)
	special_TAGs = list(itertools.product(*special_TAGs))
	print(special_TAGs)
	controls = script_name + model + datasets + score_name+ policy_mode + hyper + stage + special_TAGs
	controls = list(itertools.product(*controls))
	s = '#!/bin/bash\n'
	for i in range(len(controls)):
		s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --pin_memory --model {} --dataset {} --score_name {} --policy_mode {} --hyper {} --save_policy --policy_baseline --stage {} --special_TAG {}&\n'.format(gpu_ids[i%len(gpu_ids)],*controls[i])        
	print(s)
	run_file = open("./run.sh", "w")
	run_file.write(s)
	run_file.close()
	exit()

def generate_retrain():
	gpu_ids = ['0','1','2','3']
	script_name = [['Fine_Tuning.py']]
	model = [['resnet56']]
	datasets = [['CIFAR10']]
	score_name = [['energy']]
	policy_mode = [['ckmeans']]
	#special_TAGs = np.arange(10)
	#special_TAGs = special_TAGs.astype(str)
	#special_TAGs = list(itertools.product(*special_TAGs))
	stage = [[1,2,3,4,5,6]]
	controls = script_name + model + datasets + score_name + policy_mode + stage
	controls = list(itertools.product(*controls))
	s = '#!/bin/bash\n'
	for i in range(len(controls)):
		s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --pin_memory --model {} --dataset {} --score_name {} --policy_mode {} --stage {} --no_train&\n'.format(gpu_ids[i%len(gpu_ids)],*controls[i])        
	print(s)
	run_file = open("./run.sh", "w")
	run_file.write(s)
	run_file.close()
	exit()

def generate_flops():
	gpu_ids = ['0', '1', '2', '3']
	script_name = [['Flops_Counting.py']]
	model = [['densenet','resnet']]
	datasets = [['CIFAR10']]
	score_name = [['energy']]
	policy_mode = [['ckmeans']]
	stages = [[1]]
	controls = script_name + model + datasets + score_name+ policy_mode + stages
	controls = list(itertools.product(*controls))
	s = '#!/bin/bash\n'
	for i in range(len(controls)):
		s = s + 'CUDA_VISIBLE_DEVICES=\"{}\" python {} --pin_memory --model {} --dataset {} --score_name {} --policy_mode {} --stage {}&\n'.format(gpu_ids[i%len(gpu_ids)],*controls[i])        
	print(s)
	run_file = open("./run.sh", "w")
	run_file.write(s)
	run_file.close()
	exit()

def main():
	generate_flops()

if __name__ == '__main__':
	main()
