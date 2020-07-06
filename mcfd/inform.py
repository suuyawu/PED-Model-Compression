"""
calculate the information
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from tqdm import tqdm
from torch.utils.data.sampler import RandomSampler
import infometrics
import models
from utils import *
import config
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
def Feature_Anal_IB(model, trainLoader, param,device): #not using the whole 50000 input samples	
	for i, (images, labels) in enumerate(trainLoader):
		if i == 0:
			l_temp = labels
		else:
			l_temp = torch.cat((l_temp, labels), dim=0)
		im = images.to(device)
		Result=model(im)
		Result = Result.view(param['batchsize_score'], -1)

		if param['score_normalize']:
			Result = (Result- torch.mean(Result, dim=1, keepdim=True))/torch.std(Result, dim=1, keepdim=True)
		temp1 = Result.to('cpu') 
		temp1  = temp1.detach().numpy()
		if i == 0:
			res = temp1
		else:
			res = np.concatenate((res, temp1), 0)
			#res=torch.cat((res, Result), 0)
		if i == param['mt_batch']:
			return res, l_temp
	# res = res.to('cpu') 
	# res  = res.detach().numpy()
	# return res, l_temp 

def run(gpu,ngpus_per_node,param, name,args,blocks,device):
	if param['parallel']:
		args.gpu = gpu
		if args.gpu is not None:
			print("Use GPU: {} for training".format(args.gpu))
		args.rank = args.rank * ngpus_per_node + gpu
	seed = param['seed']
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	np.random.seed(seed)
	if param['parallel']:
		dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,world_size=args.world_size, rank=args.rank)
		args.batchsize_score = int(args.batchsize_score / ngpus_per_node)
		args.num_workers = int(args.num_workers / ngpus_per_node)
	#dist.init_process_group(backend=args.dist_backend, world_size=args.world_size, rank=args.rank)
	dataset, arch, score_name, policy_mode = name.split('_')
	if param['stage'] == 1:
		model_path = '../output/model/{}_{}.pt'.format('_'.join([dataset, arch]), 0)
		policy_arr = []
		for num_units in blocks:
			policy_arr.append([1]*num_units)
	else:
		model_path = '../output/model/{}_{}.pt'.format(name, param['stage']-1)
		policy_arr = list(np.load('../output/policy/{}_{}.npy'.format(name, param['stage']-1), allow_pickle=True))
	
	if param['score_name'] == 'fisher':
		from infometrics.FisherInfo import cal_info 
	elif param['score_name'] == 'energy':
		from infometrics.Energydist import cal_info
	elif param['score_name'] == 'shannon':
		from infometrics.MutualInfo import cal_info
	if param['resume']:
		resu_old=np.load('../output/information/{}_{}.npy'.format('_'.join([dataset, arch, score_name]), param['stage']))
		start_iter=np.load('../output/information/{}_{}_epoch.npy'.format('_'.join([dataset, arch, score_name]), param['stage']))
		score_left=param['mt_score']-start_iter
		resu_new=np.zeros((score_left, sum(blocks)-Num_deact_blc(policy_arr), 2))
		resu = np.concatenate((resu_old,resu_new),axis=0)
	else:
		start_iter=0
		resu = np.zeros((param['mt_score'], sum(blocks)-Num_deact_blc(policy_arr), 2))
		
	train_dataset = fetch_dataset(param['dataset'], split = 'train')
	if param['parallel']:
		sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
	else:
		sampler=RandomSampler(train_dataset, replacement = True)
	for e in tqdm(range(start_iter, param['mt_score'])):
		sampler.set_epoch(e)#
		k = 0
		trainLoader = load_dataset(train_dataset, batch_size=param['batchsize_score'], shuffle = False, 
			pin_memory=param['pin_memory'], num_workers=param['num_workers'],sampler = sampler)
		Err=[]
		for i in tqdm(range(len(blocks))):
			for j in range(blocks[i]):
				if policy_arr[i][j] != 0:
					model = eval('models.{}.{}(dataset = \'{}\', policy = {}, inform={}, model_path = \'{}\')'.format(param['model'],param['arch'], param['dataset'], policy_arr, [i,j],model_path))
					if param['parallel']:
						torch.cuda.set_device(args.gpu)
						model.cuda(args.gpu)
						model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
					else:
						model.to(device)
						model = torch.nn.DataParallel(model, device_ids=param['GPUs'])
					model.eval()
					res, labels = Feature_Anal_IB(model, trainLoader, param,device)
					FI_Y_Blc, err = cal_info(res, labels, param['classes_score'][param['dataset']])
					if param['score_standardize']:
						FI_Y_Blc = FI_Y_Blc / np.sqrt(res.shape[1])
					if i==0:
						resu[e, k, :] = FI_Y_Blc, j
					else:
						resu[e, k, :] = FI_Y_Blc, j+sum(blocks[i-1::-1])
					k += 1
					Err.append(err)
					if param['parallel']:
						dist.barrier()
						resu=torch.from_numpy(resu).to(device)
						dist.all_reduce(resu, op=torch.distributed.ReduceOp.SUM)
						resu=resu.to('cpu').detach().numpy()
						resu=resu/ngpus_per_node
					#print(resu[e])
					if e % (param['mt_score']//10) == 0 and gpu==0:
						#print('round {} -- block {} -- unit {} done with {}: {}.'.format(e, i, j, param['score_name'], FI_Y_Blc))
						print('round {} -- block {} -- unit {} done with {}: {}.'.format(e, i, j, param['score_name'], resu[e, k-1, 0]))
						np.save('../output/information/{}_{}.npy'.format('_'.join([dataset, arch, score_name]), param['stage']), resu)
						np.save('../output/information/{}_{}_epoch.npy'.format('_'.join([dataset, arch, score_name]), param['stage']), e+1)

def main():
	global device, blocks
	parser = config.prepare_parser()
	args=parser.parse_args()
	param = vars(parser.parse_args())
	device = torch.device(param['device'])
	blocks = param['blocks'][param['arch']]
	name = config.name_from_config(param)
	print(param, name)
	ngpus_per_node = torch.cuda.device_count()
	args.world_size = ngpus_per_node * args.world_size
	if param['parallel']:
		mp.spawn(run, nprocs=ngpus_per_node, args=(ngpus_per_node,param, name,args,blocks,device))
	else:
		run(args.gpu,ngpus_per_node,param, name,args,blocks,device)

if __name__ == '__main__':
	main()

