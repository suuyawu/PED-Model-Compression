from utils import *

import itertools
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import models
from collections import OrderedDict
import matplotlib
from matplotlib import cm
from matplotlib import pyplot as plt
	
def extracted_result(names, stages):
	extract_result = {}
	with open('../output/result/test_result.pkl', "rb", buffering=0) as f:
		results = pickle.load(f, encoding='utf-8')
	print(results)
	for name in names:
		dataset, model, score_name, policy_mode = name.split('_')
		intialname = '_'.join([dataset, model, score_name, 'ckmeans'])
		extract_result['{}'.format(name)] = {}
		extract_result['{}'.format(name)]['stage'] = [0]
		extract_result['{}'.format(name)]['acc'] = [results['{}_0'.format(intialname)]['acc']]
		extract_result['{}'.format(name)]['batch_time'] = [results['{}_0'.format(intialname)]['batch_time']]
		extract_result['{}'.format(name)]['param'] = [results['{}_0'.format(intialname)]['num_param']]
		extract_result['{}'.format(name)]['param_rate'] = [0]
		extract_result['{}'.format(name)]['flops'] = [results['{}_0'.format(intialname)]['num_flop']]
		extract_result['{}'.format(name)]['flops_rate'] = [0]
		for stage in stages:
			extract_result['{}'.format(name)]['stage'].append(stage)
			extract_result['{}'.format(name)]['acc'].append(results['{}_{}'.format(name, stage)]['acc'])
			extract_result['{}'.format(name)]['batch_time'].append(results['{}_{}'.format(name, stage)]['batch_time'])
			num_param = results['{}_{}'.format(name, stage)]['num_param']
			extract_result['{}'.format(name)]['param'].append(num_param)
			extract_result['{}'.format(name)]['param_rate'].append(1-(num_param/results['{}_0'.format(intialname)]['num_param']))
			num_flop = results['{}_{}'.format(name, stage)]['num_flop']
			extract_result['{}'.format(name)]['flops'].append(num_flop)
			extract_result['{}'.format(name)]['flops_rate'].append(1-(num_flop/results['{}_0'.format(intialname)]['num_flop']))
	return extract_result

def gather_names(models, datasets, score_names, policy_modes, policy_baseline, special_TAGs):
	controls =  datasets + models + score_names + policy_modes + policy_baseline + special_TAGs
	controls = list(itertools.product(*controls))
	model_TAG = datasets + models
	model_TAG = list(itertools.product(*model_TAG))
	names = []
	for control in controls:
		names.append('_'.join([item for item in control if not item == '']))
	return model_TAG, names

def random_barplot():
	models = [['resnet18']]
	datasets = [['tinyImageNet']]
	score_names = [['energy']]
	policy_modes = [['ckmeans']]
	policy_baseline = [['random']]
	special_TAGs = np.arange(10)
	special_TAGs = special_TAGs.astype(str)
	special_TAGs = list(itertools.product(*special_TAGs))
	model_TAG, names = gather_names(models, datasets, score_names, 
										policy_modes, policy_baseline, special_TAGs)
	stages = np.arange(6)+1
	stages = list(stages.astype(int))
	names.append('_'.join(['CIFAR10', 'resnet56', 'energy','ckmeans']))
	result = extracted_result(names, stages)
	fig_format = 'png'
	x_name = ['param_rate', 'flops_rate']
	y_name = ['acc', 'batch_time']
	model_name = ['random', 'Pruning with Energy Distance (PED) for ResNet56']
	colors = ['white','green', 'blue']
	for i, y in enumerate(y_name):
		fig1, ax1 = plt.subplots()
		data = {}
		x = [[result['{}'.format(name)][y][0] for name in names]]
		# x = np.zeros((len(stages)+1, 10))
		# x[0,:] = np.array([result['{}'.format(name)][y][0] for name in names])[0:10]
		for stage in stages:
			data[stage] = []
			for k, name in enumerate(names):
				if k == len(names)-1:
					break
				else:
					data[stage].append(result['{}'.format(name)][y][stage])
			x.append(data[stage])
			# x[stage,:] = data[stage]
		# ax1.set_title('pruning model resnet56')
		# std = x.std(axis=1)
		# x = x.mean(axis=1)
		# x1 = x+std/math.sqrt(10)
		# x2 = x-std/math.sqrt(10)
		x0 = result['{}'.format(names[len(names)-1])][y]
		line_props = dict(color='black')
		bbox_props = dict(color='black')
		flier_props = dict(marker="+")
		plt.boxplot(x, whiskerprops=line_props, boxprops=bbox_props, flierprops=flier_props)
		# plt.fill_between(np.arange(len(stages)+1), x1, x2,facecolor='0.9')
		plt.plot(np.arange(len(stages)+1)+1,x0,'g^-',label = model_name[1], color='green',markevery=100)
		# plt.plot(np.arange(len(stages)+1),x,label = model_name[0],linestyle='--',color='blue',markevery=100)
		# plt.plot(np.arange(len(stages)+1),x1,linestyle='--',color='white',markevery=100)
		# plt.plot(np.arange(len(stages)+1),x2,linestyle='--',color='white',markevery=100)
		plt.ylim((max(min(x0)-(max(x0)-min(x0)),0), max(x0)+(max(x[1])-min(x[1]))))
		# plt.xticks(np.arange(len(stages)+1), [format(comp, '.0f') for comp in result['{}'.format(names[len(names)-1])]['param_rate']])
		plt.xticks(np.arange(len(stages)+1)+1, ['initial', 'stage1','stage2','stage3', 'stage4'])
		# ax1.set_xlabel('Pruning stages', fontsize=15)
		ax1.set_ylabel('Test Accuracy', fontsize=12)
		plt.rc('legend', fontsize=10) 
		# plt.xlabel('Pruning stages')
		# plt.ylabel('Test Accuracy')
		plt.grid(False)
		plt.legend()
		plt.show()
		# fig.savefig('../output/result.{}'.format(fig_format),bbox_inches='tight',pad_inches=0)       
	plt.close()
	return

def energy_dependence():
	models = [['resnet', 'densenet']]
	datasets = [['CIFAR10', 'CIFAR100']]
	score_names = [['energy']]
	policy_modes = [['ckmeans']]
	policy_baseline = [['']]
	special_TAGs = [['']]
	model_TAG, names = gather_names(models, datasets, score_names, 
										policy_modes, policy_baseline, special_TAGs)
	stages = [0]
	ed = {}
	print(names)
	for name in names:
		ed[name] = np.load('../output/information/{}_{}.npy'.format(name, 9))
		ed[name] = np.mean(ed[name], axis=0)

	fig, ax = plt.subplots()
	# arr_ind = map(lambda x:int(x), ed[0][:,1])
	plt.hist(ed[names[2]][:,0], 50, density=False, facecolor='g', alpha=0.75, label = 'ResNet-164, stage:9, CIFAR100')
	plt.hist(ed[names[3]][:,0], 50, density=False, facecolor='b', alpha=0.75, label = 'DenseNet-100-k12, stage:9, CIFAR100')
	ax.set_xlabel('Energy Dependence', fontsize=24)
	ax.set_ylabel('Frequency', fontsize=24)
	ax.set_ylim(0,8)
	ax.set_xlim(0,600)
	plt.rc('legend', fontsize=15) 
	# ax1.set_title('Energy Dependence Estimators Histogram for CIFAR10')
	# plt.rc('ytick', labelsize=50)
	# plt.rc('xtick', labelsize=50)
	plt.legend()
	fig.savefig('../output/result/EnergyDependenceCIFAR100{}'.format(name)+'.png',bbox_inches='tight',pad_inches=0)
	return

def histogram(values, dividers):
	bins = [0 for _ in range(len(dividers)+1)]
	for num in values:
		if num > dividers[-1]:
			bins[-1] += 1
		else:
			k = 0
			while num>dividers[k]:
				k+=1
			bins[k] += 1
	return bins

def important_units():
	models = [['resnet56']]
	datasets = [['CIFAR10']]
	score_names = [['energy']]
	policy_modes = [['ckmeans']]
	policy_baseline = [['']]
	special_TAGs = [['']]
	model_TAG, names = gather_names(models, datasets, score_names, 
										policy_modes, policy_baseline, special_TAGs)
	stages = np.arange(11)+1
	units = {'resnet164':[18,18,18], 'resnet56':[9,9,9]}
	results = {}

	for name in names:
		results[name] = {}
		dataset, model, score_name, policy_mode = name.split('_')
		energydist = np.load('../output/information/{}_{}.npy'.format('_'.join([dataset, model, score_name]), 1))
		energydist = np.mean(energydist, axis=0)
		inds = energydist[:,1]
		bins = histogram(inds, [units[model][0]-1,units[model][0]+units[model][1]-1])
		results[name]['block1'] = [bins[0]]
		results[name]['block2'] = [bins[1]]
		results[name]['block3'] = [bins[2]]
		for stage in stages:
			if stage == 1:
				pass
			else:
				energydist = np.load('../output/information/{}_{}.npy'.format('_'.join([dataset, model, score_name]), stage))
				energydist = np.mean(energydist, axis=0)
				inds = energydist[:,1]
				bins = histogram(inds, [units[model][0]-1,units[model][0]+units[model][1]-1])
				results[name]['block1'].append(bins[0])
				results[name]['block2'].append(bins[1])
				results[name]['block3'].append(bins[2])
	labels = ['{}'.format(stage-1) for stage in stages]
	for name in names:
		dataset, model, score_name, policy_mode = name.split('_')

		block1 = results[name]['block1']
		block2 = results[name]['block2']
		block3 = results[name]['block3']
		fig, ax = plt.subplots()
		index = stages-1 + 0.3
		width = 0.8

		y_offset = np.zeros(len(stages))
		plt.bar(stages-1 - width*1/3, block1, width/3, bottom=y_offset, label = 'block1')
		plt.bar(stages-1 , block2, width/3, bottom=y_offset, label = 'block2')
		plt.bar(stages-1 + width*1/3, block3, width/3, bottom=y_offset, label = 'block3')
		plt.ylim((0, units[model][0]+1))
		if model == 'resnet164':
			model = 'ResNet-164'
		elif model == 'densenet':
			model = 'DenseNet-100-k12'
		elif model == 'resnet56':
			model = 'ResNet-56'
		ax.set_ylabel('The number of active units in blocks')
		ax.set_xlabel('Stages')
		ax.set_title('Pruning process for {} on {}'.format(model, dataset))
		ax.set_xticks(stages-1)
		ax.set_xticklabels(labels)
		ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
		ax.legend()
		plt.savefig('../output/result/units_{}'.format(name)+'.png',bbox_inches='tight',pad_inches=0)

	def autolabel(rects):
		"""Attach a text label above each bar in *rects*, displaying its height."""
		for rect in rects:
			height = rect.get_height()
			ax.annotate('{}'.format(height),
						xy=(rect.get_x() + rect.get_width() / 2, height),
						xytext=(0, 3),  # 3 points vertical offset
						textcoords="offset points",
						ha='center', va='bottom')

def extract_checkpoint(names, stages):
	results = extracted_result(names, stages)
	for name in names:
		# plt.rc('font', family='serif', serif='Times')
		# plt.rc('text', usetex=True)
		# plt.rc('xtick', labelsize=21)
		# plt.rc('ytick', labelsize=21)
		# plt.rc('axes', labelsize=24)

		# linewidth=4
		# marker='*'
		# markersize = 16                     
		fig, ax = plt.subplots()
		for stage in stages:
			checkpoint_path = '../output/model/'+'{}_{}_checkpoint.pth.tar'.format(name,stage)
			checkpoint = torch.load(checkpoint_path, map_location='cpu')
			losses = checkpoint['train_result']['losses']
			losses = np.array(losses)
			losses = losses.mean(axis=1)

			acc = checkpoint['test_result']
			acc = np.array(acc)
			epochs = np.arange(0, len(losses),1)
			plt.plot(epochs, acc, label='stage: {}, {:2.2%} parameters'.format(stage, 1-results['{}'.format(name)]['param_rate'][stage]))
		
		ax.set_ylabel('Test Accuracy (%)', fontsize = 24)
		ax.set_xlabel('Epochs', fontsize = 24)
		ax.set_xlim(0, len(losses)+1)
		ax.set_ylim(85,100)
		plt.legend(loc='upper left')
		plt.rc('axes', labelsize=24)
		fig.savefig('../output/result/acc_{}'.format(name)+'.png',bbox_inches='tight',pad_inches=0)
		# plt.savefig('../output/result/acc_{}'.format(name)+'.png')  
		plt.clf()

def train_process():
	models = [['resnet56']]
	datasets = [['CIFAR10']]
	score_names = [['energy']]
	policy_modes = [['ckmeans']]
	policy_baseline = [['']]
	special_TAGs = [['']]
	model_TAG, names = gather_names(models, datasets, score_names, 
										policy_modes, policy_baseline, special_TAGs)
	stages = np.arange(6)+1
	stages = stages.astype(int)

	extract_checkpoint(names, stages)
	

def stagewise_results():
	models = [['resnet56']]
	datasets = [['CIFAR10']]
	score_names = [['energy']]
	policy_modes = [['ckmeans', 'random', 'rank']]
	policy_baseline = [['']]
	special_TAGs = [['']]
	model_TAG, names = gather_names(models, datasets, score_names, 
										policy_modes, policy_baseline, special_TAGs)
	print(names)
	stages = np.arange(11)+1
	stages = stages.astype(int)
	results = extracted_result(names, stages)
	import pandas as pd
	for key in results.keys():
		df = pd.DataFrame(results[key].values(), results[key].keys()) 
		df = df.T.to_latex(index=False)
		print(df)

def main():
	important_units()

if __name__ == '__main__':
	main()