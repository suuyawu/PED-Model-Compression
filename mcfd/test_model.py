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

parser = config.prepare_parser()
param = vars(parser.parse_args())
device = torch.device(param['device'])

def test(data_loader, model, criterion):
	batch_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5= AverageMeter()
	# switch to evaluate mode
	model.eval()
	start_time=time.time()
	end = time.time()
	for i, (input, target) in enumerate(data_loader):
		input = input.to(device)
		target = target.to(device)

		# compute output
		output = model(input)
		loss = criterion(output, target)

		# measure accuracy and record loss
		prec = Accuracy(output, target, topk=(1,5))
		prec1=prec[0]
		prec5=prec[1]
		losses.update(loss.item(), input.size(0))
		top1.update(prec1.item(), input.size(0))
		top5.update(prec5.item(), input.size(0))
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % 50 == 0:
			print('Test: [{0}/{1}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
					  i, len(data_loader), batch_time=batch_time, loss=losses,
					  top1=top1,top5=top5))

	print('*Prec@1 {top1.avg:.3f}'.format(top1=top1))
	print('*Prec@5 {top5.avg:.3f}'.format(top5=top5))
	end_time=time.time()
	time_all=end_time-start_time
	print('Total time for one test run is {}s'.format(time_all))
	return top1.avg, top5.avg, batch_time.avg

def run(param, name):
	seed = param['seed']
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)

	if param['stage'] == 0:
		dataset, arch, score_name, policy_mode = name.split('_')
		model_path = '../output/model/{}_{}.pt'.format('_'.join([dataset, arch]), 0)
		policy = []
		for num_units in param['blocks'][param['arch']]:
			policy.append([1]*num_units)
	else:
		if not param['special_TAG'] == '':
			name = '_'.join([
				name,
				param['special_TAG']])
			seed = seed + int(param['special_TAG'])
		model_path = '../output/model/{}_{}.pt'.format(name, param['stage'])
		policy = np.load('../output/policy/{}_{}.npy'.format(name, param['stage']), allow_pickle=True)
		policy = policy.tolist()
	model = eval('models.{}.{}(dataset = \'{}\', policy = {}, model_path = \'{}\').to(device)'.format(param['model'],param['arch'], param['dataset'], policy, model_path))
	torch.save(model,'../output/model/{}_{}.pth'.format(name, param['stage']))
	model = nn.DataParallel(model, device_ids=param['GPUs']) if param['parallel'] else model
	test_dataset = fetch_dataset(param['dataset'], split = 'test')
	test_loader = load_dataset(test_dataset, param['batch_size']['test'], 
			param['shuffle']['test'], param['pin_memory'], param['num_workers'])
	criterion = nn.CrossEntropyLoss()
	prec1, prec5, batch_time = test(test_loader, model, criterion)
	#num_param = Param_cnt(model)
	num_flop, num_param = Flops_cnt(model, param, device)
	test_result = {'acc@1':prec1,'acc@5':prec5, 'batch_time': batch_time, 'num_param':num_param,'num_flop':num_flop}
	print(test_result)
	if not path.exists('../output/result/test_result.pkl'):
		results = {}
	else:
		with open('../output/result/test_result.pkl', "rb", buffering=0) as f:
			results = pickle.load(f, encoding='utf-8')
	results['{}_{}'.format(name, param['stage'])] = test_result
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
