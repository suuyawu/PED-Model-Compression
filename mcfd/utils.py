"""
utils file
"""
from __future__ import print_function
import sys
import os
import numpy as np
import time
import pickle
import operator
from functools import reduce
from PIL import Image

from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision import datasets as dsets
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
# from copy import deepcopy

class Lighting(object):
	def __init__(self, alphastd, eigval, eigvec):
		self.alphastd = alphastd
		self.eigval = eigval
		self.eigvec = eigvec
	def lighting(self, img, alphastd, eigval, eigvec):
		if not transforms.functional._is_pil_image(img):
			raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
		img = transforms.functional.to_tensor(img)
		alpha = img.new().resize_(3).normal_(0, alphastd)
		rgb = eigvec.type_as(img).clone()\
			.mul(alpha.view(1, 3).expand(3, 3))\
			.mul(eigval.view(1, 3).expand(3, 3))\
			.sum(1).squeeze()
		return transforms.functional.to_pil_image(img.add(rgb.view(3, 1, 1).expand_as(img)))
	def __call__(self, img):
		if self.alphastd == 0:
			return img
		return self.lighting(img, self.alphastd, self.eigval, self.eigvec)

class Cutout(object):
	def __init__(self, length):
		self.length = length

	def __call__(self, img):
		h, w = img.size(1), img.size(2)
		mask = np.ones((h, w), np.float32)
		y = np.random.randint(h)
		x = np.random.randint(w)

		y1 = np.clip(y - self.length // 2, 0, h)
		y2 = np.clip(y + self.length // 2, 0, h)
		x1 = np.clip(x - self.length // 2, 0, w)
		x2 = np.clip(x + self.length // 2, 0, w)

		mask[y1: y2, x1: x2] = 0.
		mask = torch.from_numpy(mask)
		mask = mask.expand_as(img)
		img *= mask
		return img

class Iter_data(object):
	def __init__(self, dataset, transform):
		self.index = 0
		self.source = dataset
		self.source_trans = torch.stack([transform(x) for x in self.source.data])
		self.len = len(self.source.data)
		self.perm = torch.randperm(self.len).tolist()

	def next_item(self, batch_size):
		indd = self.perm[self.index*batch_size:(self.index+1)*batch_size]
		image = self.source_trans[indd]
		image = image.float() 
		label = torch.from_numpy(np.array(self.source.targets))
		label = label[indd]
		self.index += 1
		if self.index == int(self.len//batch_size):
			self.index = 0
		yield (image, label)

### data
def fetch_dataset(data_name, split = 'train', normalize = True, cutout = False):
	print('loading data {}...'.format(data_name))
	if(data_name=='CIFAR10'):
		if(normalize):
			normMean = [0.49139968, 0.48215827, 0.44653124]
			normStd = [0.24703233, 0.24348505, 0.26158768]
			trainTransform = transforms.Compose([transforms.RandomCrop(32, padding=4),
											transforms.RandomHorizontalFlip(),
											transforms.ToTensor(),
											transforms.Normalize(normMean, normStd)])
			if cutout:
				trainTransform.transforms.append(Cutout(16))
			testTransform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(normMean, normStd)])
		else:
			trainTransform = transforms.Compose([transforms.RandomCrop(32, padding=4),
											transforms.RandomHorizontalFlip(),
											transforms.ToTensor()])
			testTransform = transforms.Compose([transforms.ToTensor()])
		if split == 'train':
			dataset = dsets.CIFAR10(root='../data/{}/{}'.format(data_name, 'train'), train=True, download=True, transform=trainTransform)            
		elif split == 'val':
			dataset = dsets.CIFAR10(root='../data/{}/{}'.format(data_name, 'train'), train=True, download=True, transform=testTransform)
		else:
			dataset = dsets.CIFAR10(root='../data/{}/{}'.format(data_name, 'test'), train=False, download=True, transform=testTransform)
	elif(data_name=='CIFAR100'):
		if(normalize):
			normMean = [0.49139968, 0.48215827, 0.44653124]
			normStd = [0.24703233, 0.24348505, 0.26158768]
			trainTransform = transforms.Compose([transforms.RandomCrop(32, padding=4),
											 transforms.RandomHorizontalFlip(),
											 transforms.ToTensor(),
											 transforms.Normalize(normMean, normStd)])
			if cutout:
				trainTransform.transforms.append(Cutout(8))
			testTransform = transforms.Compose([transforms.ToTensor(), 
				transforms.Normalize(normMean, normStd)])
		else:
			trainTransform = transforms.Compose([transforms.RandomCrop(32, padding=4),
											 transforms.RandomHorizontalFlip(),
											 transforms.ToTensor()])
			testTransform = transforms.Compose([transforms.ToTensor()])
		if split == 'train':
			dataset = dsets.CIFAR100(root='../data/{}/{}'.format(data_name, 'train'), train=True, download=True, transform=trainTransform)
		elif split == 'val':
			dataset = dsets.CIFAR100(root='../data/{}/{}'.format(data_name, 'train'), train=True, download=True, transform=testTransform)
		else:
			dataset = dsets.CIFAR100(root='../data/{}/{}'.format(data_name, 'test'), train=False, download=True, transform=testTransform)
	elif(data_name=='SVHN'):
		if(normalize):
			normMean = [0.4309, 0.4302, 0.4463]
			normStd = [0.1253, 0.1282, 0.1147]
			trainTransform = transforms.Compose([
											transforms.Pad(4),
											transforms.RandomCrop(32),
											transforms.RandomHorizontalFlip(),
											transforms.ToTensor(),
											transforms.Normalize(normMean,normStd)])
			testTransform = transforms.Compose([transforms.ToTensor(),
											transforms.Normalize(normMean,normStd)])
		else:
			trainTransform = transforms.Compose([transforms.ToTensor()])
			testTransform = transforms.Compose([transforms.ToTensor()])
		if split == 'train':
			traindataset = dsets.SVHN(root='../data/{}/{}'.format(data_name, 'train'), split='train', download=True, transform = trainTransform)
			extradataset = dsets.SVHN(root='../data/{}/{}'.format(data_name, 'train'), split='extra', download=True, transform = trainTransform)
			dataset = torch.utils.data.ConcatDataset([traindataset, extradataset])
		elif split == 'val':
			traindataset = dsets.SVHN(root='../data/{}/{}'.format(data_name, 'train'), split='train', download=True, transform = testTransform)
			extradataset = dsets.SVHN(root='../data/{}/{}'.format(data_name, 'train'), split='extra', download=True, transform = testTransform)
			dataset = torch.utils.data.ConcatDataset([traindataset, extradataset])
		else:
			dataset = dsets.SVHN(root='../data/{}/{}'.format(data_name, 'test'), split='test', download=True, transform = testTransform)
	elif(data_name == 'tinyImageNet'):
		if(normalize):
			normMean = [0.4802, 0.4481, 0.3975]
			normStd = [0.2302, 0.2265, 0.2262]
			trainTransform = transforms.Compose([
				transforms.RandomRotation(20),
				transforms.RandomHorizontalFlip(0.5),
				transforms.ToTensor(),
				#transforms.Normalize(normMean, normStd),
				])
			testTransform = transforms.Compose([transforms.ToTensor(),
				#transforms.Normalize(normMean, normStd),
				])
		data_dir = '../data/tiny-imagenet-200/'
		if split == 'train':
			dataset = dsets.ImageFolder(os.path.join(data_dir, 'train'), trainTransform)
		elif split == 'val':
			dataset = dsets.ImageFolder(os.path.join(data_dir, 'train'), testTransform)
		else:
			dataset = dsets.ImageFolder(os.path.join(data_dir, 'test'), testTransform)
			# dataset = dsets.ImageFolder(os.path.join(data_dir, 'val'), testTransform)
	elif(data_name == 'ImageNet'):
		if(normalize):
			normMean = [0.485, 0.456, 0.406]
			normStd = [0.229, 0.224, 0.225]
			pca_eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
			pca_eigvec = torch.Tensor([[-0.5675, 0.7192, 0.4009],
										[-0.5808, -0.0045, -0.8140],
										[-0.5836, -0.6948, 0.4203], ])	
			trainTransform = transforms.Compose([transforms.RandomResizedCrop(224),
				#transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
				#Lighting(0.1, pca_eigval, pca_eigvec),
				transforms.RandomHorizontalFlip(),
				transforms.ToTensor(),
				transforms.Normalize(normMean, normStd)])
			if cutout:
				trainTransform.transforms.append(Cutout(56))
			testTransform = transforms.Compose([
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize(normMean, normStd)
				])
		if split == 'train':
			dataset = dsets.ImageNet(root='../data/{}'.format(data_name), split='train', download=None, transform=trainTransform)
		elif split == 'val':
			dataset = dsets.ImageNet(root='../data/{}'.format(data_name), split='train', download=None, transform=testTransform)
		else:
			dataset = dsets.ImageNet(root='../data/{}'.format(data_name), split='val', download=None, transform=testTransform)
	else:
		raise ValueError('Not valid dataset name')
	# if Iter:
	# 	trainTransform.transforms.insert(0, transforms.ToPILImage(mode=None))
	# 	train_dataset.transform = trainTransform
	# 	train_loader = Iter_data(train_dataset, trainTransform)
	# 	return train_loader
	# if valid:
	# 	train_sampler, valid_sampler = validate_dataset(train_dataset, train_dataset)
	# 	return train_dataset, validation_dataset, train_sampler, valid_sampler
	# else:
	return dataset

def load_dataset(dataset, batch_size, shuffle, pin_memory, num_workers, sampler = None):
	data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
			pin_memory=pin_memory, sampler=sampler, num_workers=num_workers)
	return data_loader

def validate_dataset(train_dataset, p=0.9):
	indices = list(range(len(train_dataset)))
	split = int(np.floor(p * len(train_dataset)))
	np.random.shuffle(indices)
	valid_idx, train_idx = indices[split:], indices[:split]
	train_sampler = SubsetRandomSampler(train_idx)
	valid_sampler = SubsetRandomSampler(valid_idx)
	return train_sampler, valid_sampler

def Num_deact_blc(policy):
	temp = 0
	for item in policy:
		item = np.array(item)
		temp += len(item) - np.sum(item)
	return temp    

def digitize(x, xn, bins, ind_arr):
	dic = {}
	for i,val in enumerate(bins):
		for j, val_temp in enumerate(x):
			if val_temp>=bins[i] and val_temp<bins[i+1]:
				if val not in dic:
					dic[val] = [ind_arr[j]]
				else:
					dic[val].append(ind_arr[j])
		if i == len(bins)-2:
			break
	u = np.max(bins)
	if u in dic:
		dic[u].append(ind_arr[np.argmax(xn)])
	else:
		dic[u] = [ind_arr[np.argmax(xn)]]
	return dic

def Get_index_in_Policy(x, policy):
	l1 = int(len(policy[0]))
	l2 = int(len(policy[1]))
	l3 = int(len(policy[2]))
	for it in x:
		it = int(round(it))
		if it < l1:
			policy[0][it] = 0
		elif (it >= l1) and (it < l1+l2):
			policy[1][it -l1] = 0
		elif (it >= l1+l2)  and (it < l1+l2+l3):
			policy[2][it-l1-l2] = 0
		else:
			policy[3][it-l1-l2-l3] = 0
	return policy

def Policy_random(policy0, policy0_index, num):
	policy = np.copy(policy0)
	num0 = Num_deact_blc(policy0)
	randn = np.random.choice(policy0_index, num-num0, replace=False).astype(int)
	l1 = int(len(policy[0]))
	l2 = int(len(policy[1]))
	for val in randn:
		if val < l1:
			policy[0][val] = 0
		elif (val >= l1) and (val < l1+l2):
			policy[1][val-l1] = 0
		else:
			policy[2][val-l1-l2] = 0
	return policy

def Policy_rank(policy0, policy0_index, num, blc):
	policy = np.copy(policy0)
	num0 = Num_deact_blc(policy0)
	sortidx = policy0_index[blc.argsort()]
	rank = sortidx[:(num-num0)]
	l1 = int(len(policy[0]))
	l2 = int(len(policy[1]))
	for val in rank:
		if val < l1:
			policy[0][val] = 0
		elif (val >= l1) and (val < l1+l2):
			policy[1][val-l1] = 0
		else:
			policy[2][val-l1-l2] = 0
	return policy

def Policy(policy0, MI_blc_out, hyper, config):
	blc = np.copy(MI_blc_out[:,0])
	policy0_index = np.nonzero(np.asarray(policy0).reshape(-1))[0]
	if config['policy_mode'] == 'random':
		num_of_deactivate_blc = int(config['hyper'][0])
		policy = Policy_random(policy0, policy0_index, num_of_deactivate_blc)
	elif config['policy_mode'] == 'rank':
		num_of_deactivate_blc = int(config['hyper'][0])
		policy = Policy_rank(policy0, policy0_index, num_of_deactivate_blc, blc)
	elif config['policy_mode'] == 'ckmeans':
		policy = np.copy(policy0)
		from ckmeans import ckmeans
		groups = ckmeans(blc, int(hyper))
		digit = {}
		for k, vals in enumerate(groups):
			index = np.where(np.in1d(MI_blc_out[:,0], vals))[0]
			digit[k] = np.take(MI_blc_out[:,1], index)
		for key,val in digit.items():
			if len(val) > config['num_halting']:
				if not config['back']:
					policy = Get_index_in_Policy(val[config['num_halting']:], policy)
				else:
					policy = Get_index_in_Policy(val[0:(len(val)-config['num_halting'])], policy)
	else:
		raise ValueError('Not valid policy mode')
	num_of_deactivate_blc = Num_deact_blc(policy)
	return policy.tolist(), num_of_deactivate_blc
	
def Param_cnt(model):
	return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])

def Flops_cnt(model, param, device):
	from thop import profile
	if param['dataset'] == 'CIFAR10':
		dsize = (1, 3, 32, 32)
	elif param['dataset'] == 'CIFAR100':
		dsize = (1, 3, 32, 32)
	elif param['dataset'] == 'SVHN':
		dsize = (1, 3, 32, 32)
	elif param['dataset'] == 'tinyImageNet':
		dsize = (1, 3, 64, 64)
	elif param['dataset'] == 'ImageNet':
		dsize = (1, 3, 224, 224)
	input = torch.randn(dsize).to(device)
	macs, params = profile(model, inputs=(input, ))
	return macs, params

def Accuracy(output, target, topk=(1,5)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

## Evaluation Metrics
class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
		self.history = []

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count
		self.history.append(self.val)

def save_checkpoint(state, is_best, name, stage):
	directory = '../output/model/'
	if not os.path.exists(directory):
		os.makedirs(directory)

	checkpoint_path = directory + '{}_{}_checkpoint.pth.tar'.format(name,stage)
	state_path = directory + '{}_{}.pt'.format(name,stage)
	torch.save(state, checkpoint_path)
	if is_best:
		best_path = directory + '{}_{}_best.pth.tar'.format(name,stage)
		torch.save(state, best_path)
		torch.save(state['state_dict'], state_path)