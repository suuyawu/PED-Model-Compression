"""
Retrain the pruned model with active units
"""
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import OrderedDict

import config
import models
from utils import *
from test_model import test

def train(train_loader, model, criterion, optimizer, epoch):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	model.train()
	end = time.time()
	start_time=time.time()
	for i, (input, target) in enumerate(train_loader):
		input = input.to(device)
		target = target.to(device)

		output = model(input)
		loss = criterion(output, target)
		loss = torch.mean(loss) if parallel else loss

		prec = Accuracy(output, target, topk=(1,5))
		prec1=prec[0]
		prec5=prec[1]
		losses.update(loss.item(), input.size(0))
		top1.update(prec1.item(), input.size(0))
		top5.update(prec5.item(), input.size(0))

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		batch_time.update(time.time() - end)
		end = time.time()

		if i % 100 == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
					  epoch, i, len(train_loader), batch_time=batch_time,
					  loss=losses, top1=top1,top5=top5))
	end_time=time.time()
	time_all=end_time-start_time
	print('Total time for one train run is {}s'.format(time_all))
	return losses.history, top1.history

def train_model(data_loader, model, criterion, optimizer, scheduler, start_iter, epochs, name, best_prec1=0, best_prec5=0):
	best_prec1 = best_prec1
	best_prec5 = best_prec5
	loss = []
	train_acc = []
	test_acc = []
	for epoch in range(start_iter,epochs):
		start_time=time.time()
		lossbatch, accbatch = train(data_loader['train'], model, criterion, optimizer, epoch)
		loss.append(lossbatch)
		train_acc.append(accbatch)
		prec1, prec5,_ = test(data_loader['eval'], model, criterion)
		scheduler.step()

		is_best = prec1 > best_prec1
		best_prec1 = max(prec1, best_prec1)
		test_acc.append(best_prec1)
		best_prec5 = max(prec5, best_prec5)
		test_acc.append(best_prec5)
		end_time=time.time()
		time_all=end_time-start_time
		
		save_checkpoint({
			'epoch': epoch + 1,
			'state_dict': model.module.state_dict() if parallel else model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'scheduler_state_dict': scheduler.state_dict(),
			'train_result': {'losses':loss, 'train_acc':train_acc},
			'test_result': test_acc,
			'best_prec1': best_prec1,
			'best_prec5': best_prec5,
			'epoch_time':time_all
		}, is_best, name, stage)
		print('Total time for one epoch is {}s'.format(time_all))
	print('Best accuracy@1: ', best_prec1)
	print('Best accuracy@5: ', best_prec5)
	

def make_optimizer(model, param):
    if(param['optimizer_name']=='Adam'):
        optimizer = optim.Adam(model.parameters(),lr=param['lr'])
    elif(param['optimizer_name']=='SGD'):
        optimizer = optim.SGD(model.parameters(),lr=param['lr'], momentum=0.9, weight_decay=param['weight_decay'])
    else:
        raise ValueError('Optimizer name not supported')
    return optimizer
    
def make_scheduler(optimizer, param):
    if(param['scheduler_name']=='MultiStepLR'):
        scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=param['milestones'],gamma=0.1)
    elif(param['scheduler_name']=='StepLR'):
    	scheduler = lr_scheduler.StepLR(optimizer, step_size=param['step_size'], gamma=0.1)
    else:
        raise ValueError('Scheduler_name name not supported')
    return scheduler

def run(param, name):
	seed = param['seed']
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)

	if stage == 1:
		dataset, arch, _, _ = name.split('_')
		model_path = '../output/model/{}_{}.pt'.format('_'.join([dataset, arch]), 0)
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
		model_path = '../output/model/{}_{}.pt'.format(name, stage-1)

	if param['validation']:
		train_dataset = fetch_dataset(param['dataset'], split = 'train')
		validation_dataset = fetch_dataset(param['dataset'], split = 'val')
		train_sampler, valid_sampler = validate_dataset(train_dataset)

		valid_loader = load_dataset(validation_dataset, param['batch_size']['test'], 
			param['shuffle']['test'], param['pin_memory'], param['num_workers'], sampler = valid_sampler)
		train_loader = load_dataset(train_dataset, param['batch_size']['train'], 
			param['shuffle']['train'], param['pin_memory'], param['num_workers'], sampler = train_sampler)
		data_loader = {'train': train_loader, 'eval': valid_loader} 
	else:
		train_dataset = fetch_dataset(param['dataset'], split = 'train')
		test_dataset = fetch_dataset(param['dataset'], split = 'test')

		train_loader = load_dataset(train_dataset, param['batch_size']['train'], 
			param['shuffle']['train'], param['pin_memory'], param['num_workers'])
		test_loader = load_dataset(test_dataset, param['batch_size']['test'], 
			param['shuffle']['test'], param['pin_memory'], param['num_workers'])
		data_loader = {'train': train_loader, 'eval': test_loader}  
	
	policy = np.load('../output/policy/{}_{}.npy'.format(name, stage), allow_pickle=True)
	policy = policy.tolist()
	model = eval('models.{}.{}(dataset = \'{}\', policy = {}, model_path = \'{}\').to(device)'.format(param['model'],param['arch'], param['dataset'], policy, model_path))
	model = nn.DataParallel(model, device_ids=param['GPUs']) if param['parallel'] else model

	criterion = nn.CrossEntropyLoss()
	optimizer = make_optimizer(model, param)
	scheduler = make_scheduler(optimizer, param)
	
	start_iter=0
	if param['resume']:
		checkpoint_path = '../output/model/{}_{}_checkpoint.pth.tar'.format(name, stage)
		checkpoint = torch.load(checkpoint_path)
		if param['parallel']:
			new_state_dict = OrderedDict()
			for k, v in checkpoint['state_dict'].items():
				save_name='module.'+k
				new_state_dict[save_name]=v
			model.load_state_dict(new_state_dict)
		else:
			model.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		start_iter = checkpoint['epoch']

	train_model(data_loader, model, criterion, optimizer, scheduler, start_iter, param['epochs'], name)

def main():
	global device, parallel, stage
	parser = config.prepare_parser()
	param = vars(parser.parse_args())
	device = torch.device(param['device'])
	parallel = param['parallel']
	stage = param['stage']
	name = config.name_from_config(param)
	print(param, name)
	run(param, name)

if __name__ == '__main__':
	main()  
