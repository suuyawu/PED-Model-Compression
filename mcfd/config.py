''' config file
'''

from argparse import ArgumentParser

def prepare_parser():
	usage = 'Parser for all scripts.'
	parser = ArgumentParser(description=usage)

	parser.add_argument(
		'--seed', type=int, default=0,
		help='Random seed to use for initialization'
				 '(default: %(default)s)')
	parser.add_argument(
		'--device', type=str, default='cuda',
		help='GPU or CPU, out of cuda,cpu.'
				 '(default: %(default)s)')
	parser.add_argument(
		'--parallel', action = 'store_true', default=False,
		help='GPU parallel or not.'
				 '(default: %(default)s)')
	parser.add_argument(
		'--GPUs', nargs = '+', type=list, default=[0,1,2,3],
		help='the id of GPUs.'
				 '(default: %(default)s)')
	### Dataset/Dataloader stuff ###
	parser.add_argument(
		'--dataset', type=str, default='CIFAR10',
		help='Which Dataset to train on, out of CIFAR10, CIFAR100, SVHN, tinyImageNet, ImageNet.'
				 '(default: %(default)s)')
	parser.add_argument(
		'--num_workers', type=int, default=0,
		help='Number of dataloader workers.'
				 '(default: %(default)s)')
	parser.add_argument(
		'--pin_memory', action='store_true', dest='pin_memory', default=False,
		help='Pin data into memory through dataloader.' 
				 '(default: %(default)s)')
	parser.add_argument(
		'--shuffle', type=dict, default={'train':True,'test':False},
		help='Shuffle the data. '
			     '(default: %(default)s)')
	parser.add_argument(
		'--validation', action='store_true', default=False,
		help='seperate validation during pruning?'
				 '(default: %(default)s)')
	### Stagewise policy selection stuff ###
	parser.add_argument(
		'--stage', type=int, default=1,
		help='Default stage for policy selection'
				 '(default: %(default)s)')
	
	parser.add_argument(
		'--save_policy', action='store_true', default=False,
		help='Default save policy or not for policy selected'
				 '(default: %(default)s)')
	parser.add_argument(
		'--policy_mode', type=str, default='ckmeans',
		help='how to select policy, out of ckmeans, random, rank'
				 '(default: %(default)s)')
	parser.add_argument(
		'--back', action='store_true', default=False,
		help='delete the units close to output or not'
				 '(default: %(default)s)')
	parser.add_argument(
		'--resume', action='store_true', default=False,
		help='resume'
				 '(default: %(default)s)')
	parser.add_argument(
		'--hyper', nargs = '+', type=float, default=[30, 35, 40, 45],
		help='hyperparameter selections for, (1)number of clusters for ckmenas; (2)num of units dropped for randomly dropping or rankly dropping'
				 '(default: %(default)s)')

	parser.add_argument(
		'--score_name', type=str, default='energy',
		help='how to select policy, out of fisher, energy, shannon'
				 '(default: %(default)s)')
	parser.add_argument(
		'--score_normalize', action='store_true', default=False,
		help='nomalize score or not, set as true when score name is fisher'
				 '(default: %(default)s)')
	parser.add_argument(
		'--score_standardize', action='store_true', default=False,
		help='standardize score or not, set as true when score name is energy'
				 '(default: %(default)s)')
	### Model stuff ###
	parser.add_argument(
		'--model', type=str, default='densenet',
		help='model structure, out of densenet, resnet'
				 '(default: %(default)s)')
	parser.add_argument(
		'--arch', type=str, default='densenet100',
		help='model architecture to be compressed, out of densenet100, densenet121, resnet164, resnet56, resnet18'
				 '(default: %(default)s)')
	parser.add_argument(
		'--blocks', type=dict, default={'densenet100':[16, 16, 16], 'densenet121':[6, 12, 24, 16],
										'resnet164':[18, 18, 18], 'resnet56':[9, 9, 9], 'resnet18':[2, 2, 2, 2],'resnet50':[3, 4, 6, 3]},
		help='units settings, [16,16,16] for densenet, [18,18,18] for resnet164, [9,9,9] for resnet56'
				 '(default: %(default)s)')
	### Score calculation ###
	parser.add_argument(
		'--mt_score', type=int, default=20,
		help='num of rounds when compute Fisher Divergence, Energy dependence and Mutual Information'
				 '(default: %(default)s)')
	parser.add_argument(
		'--mt_batch', type=int, default=80,
		help='num of batches when compute Fisher Divergence, Energy dependence and Mutual Information'
				 '(default: %(default)s)')
	parser.add_argument(
		'--batchsize_score', type=int, default=64,
		help='batchsize for computing Fisher Divergence or Energy distance'
				 '(default: %(default)s)')
	parser.add_argument(
		'--classes_score', type=dict, default={'CIFAR10':10, 'SVHN':10, 'CIFAR100':100, 'tinyImageNet':200, 'ImageNet':1000},
		help='num of classes for computing Fisher Divergence or Energy distance'
				 '(default: %(default)s)')
	parser.add_argument(
		'--num_halting', type=int, default=1,
		help='num of policy select each time'
				 '(default: %(default)s)')
	### training stuff ###
	parser.add_argument(
		'--batch_size', type=dict, default={'train':256,'test':128},
		help='Default batchsize for DataLoader'
				 '(default: %(default)s)')
	parser.add_argument(
		'--epochs', type=int, default=100,
		help='Number of epochs to retrain model for'
				 '(default: %(default)s)')
	parser.add_argument(
		'--lr', type=float, default=0.1,
		help='learning rate, select by fine tuning'
				 '(default: %(default)s)')
	parser.add_argument(
		'--scheduler_name', type=str, default='StepLR',
		help='scheduler name'
				 '(default: %(default)s)')
	parser.add_argument(
		'--optimizer_name', type=str, default='SGD',
		help='optimizer name'
				 '(default: %(default)s)')
	parser.add_argument(
		'--step_size', type=int, default=25,
		help='step_size, increase during re-training' 
				 '(default: %(default)s)')
	parser.add_argument(
		'--milestones', type=int, default=[40,60],
		help='step_size, increase during re-training' 
				 '(default: %(default)s)')
	parser.add_argument(
		'--weight_decay', type=float, default=1e-4,
		help='weight_decay, select by fine tuning'
				 '(default: %(default)s)')
	### Special TAG ###
	parser.add_argument(
		'--special_TAG', type=str, default='',
		help='special tag for random runs'
				 ' (default: %(default)s)')
	parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
	parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
	parser.add_argument('--dist-url', default='tcp://0.0.0.0:29500', type=str,
                    help='url used to set up distributed training')
	parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
	parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
	return parser

# Name an experiment based on its config
def name_from_config(param):
	name = '_'.join([
		item for item in [
			param['dataset'],
			param['arch'],
			param['score_name'],
			param['policy_mode']]])
	return name
