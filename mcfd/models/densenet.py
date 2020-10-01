'''
The Pytorch Implementation of Pruning-version DenseNet-s Architecture, as described in paper [1]. 
This implementation is modifed based on the Pytorch released source in [2].

Reference:
[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
    Densely Connected Convolutional Networks. arXiv:1608.06993.
[2] https://pytorch.org/hub/pytorch_vision_densenet/
'''
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
# from .utils import load_state_dict_from_url
from torch import Tensor
from torch.jit.annotations import List

__all__ = ['DenseNet', 'densenet121']

device = torch.device('cuda')

class _DenseLayer(nn.Module):
	def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
		super(_DenseLayer, self).__init__()
		self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
		self.add_module('relu1', nn.ReLU(inplace=True)),
		self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
										   growth_rate, kernel_size=1, stride=1,
										   bias=False)),
		self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
		self.add_module('relu2', nn.ReLU(inplace=True)),
		self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
										   kernel_size=3, stride=1, padding=1,
										   bias=False)),
		self.drop_rate = float(drop_rate)
		self.memory_efficient = memory_efficient

	def bn_function(self, inputs):
		# type: (List[Tensor]) -> Tensor
		concated_features = torch.cat(inputs, 1)
		bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
		return bottleneck_output

	# todo: rewrite when torchscript supports any
	def any_requires_grad(self, input):
		# type: (List[Tensor]) -> bool
		for tensor in input:
			if tensor.requires_grad:
				return True
		return False

	@torch.jit.unused  # noqa: T484
	def call_checkpoint_bottleneck(self, input):
		# type: (List[Tensor]) -> Tensor
		def closure(*inputs):
			return self.bn_function(*inputs)

		return cp.checkpoint(closure, input)

	@torch.jit._overload_method  # noqa: F811
	def forward(self, input):
		# type: (List[Tensor]) -> (Tensor)
		pass

	@torch.jit._overload_method  # noqa: F811
	def forward(self, input):
		# type: (Tensor) -> (Tensor)
		pass

	# torchscript does not yet support *args, so we overload method
	# allowing it to take either a List[Tensor] or single Tensor
	def forward(self, input):  # noqa: F811
		if isinstance(input, Tensor):
			prev_features = [input]
		else:
			prev_features = input

		if self.memory_efficient and self.any_requires_grad(prev_features):
			if torch.jit.is_scripting():
				raise Exception("Memory Efficient not supported in JIT")

			bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
		else:
			bottleneck_output = self.bn_function(prev_features)

		new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
		if self.drop_rate > 0:
			new_features = F.dropout(new_features, p=self.drop_rate,
									 training=self.training)

		return new_features


class _DenseBlock(nn.ModuleDict):
	_version = 2

	def __init__(self, num_layers, num_input_features, policy, bn_size, growth_rate, drop_rate, memory_efficient=False):
		super(_DenseBlock, self).__init__()
		for i in range(num_layers):
			if policy[i]:
				layer = _DenseLayer(
					num_input_features + i * growth_rate,
					growth_rate=growth_rate,
					bn_size=bn_size,
					drop_rate=drop_rate,
					memory_efficient=memory_efficient,
				)
			else:
				layer = nn.Identity()
			self.add_module('denselayer%d' % (i + 1), layer)
		self.policy = policy
		self.growth_rate = growth_rate

	def forward(self, init_features):
		features = [init_features]
		for i, (name, layer) in enumerate(self.items()):
			features_temp = torch.cat(features, 1)
			if self.policy[i]:
				new_features = layer(features)
			else:
				new_features = torch.zeros((features_temp.shape[0], self.growth_rate, features_temp.shape[2], features_temp.shape[3])).to(device)
			features.append(new_features)
		return torch.cat(features, 1)


class _Transition(nn.Sequential):
	def __init__(self, num_input_features, num_output_features):
		super(_Transition, self).__init__()
		self.add_module('norm', nn.BatchNorm2d(num_input_features))
		self.add_module('relu', nn.ReLU(inplace=True))
		self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
										  kernel_size=1, stride=1, bias=False))
		self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
	"""Densenet-BC model class, based on
	`"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
	Args:
		growth_rate (int) - how many filters to add each layer (`k` in paper)
		block_config (list of 4 ints) - how many layers in each pooling block
		num_init_features (int) - the number of filters to learn in the first convolution layer
		bn_size (int) - multiplicative factor for number of bottle neck layers
		  (i.e. bn_size * k features in the bottleneck layer)
		drop_rate (float) - dropout rate after each dense layer
		num_classes (int) - number of classification classes
		memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
		  but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
	"""

	def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
				 num_init_features=64, policy=[[1]*6,[1]*12,[1]*24,[1]*16], inform, bn_size=4, drop_rate=0, num_classes=1000, memory_efficient=False):
		super(DenseNet, self).__init__()

		# First convolution
		self.features = nn.Sequential(OrderedDict([
			('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
								padding=3, bias=False)),
			('norm0', nn.BatchNorm2d(num_init_features)),
			('relu0', nn.ReLU(inplace=True)),
			('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
		]))

		# Each denseblock
		self.inform=inform
		self.block_config=block_config
		self.policy=policy
		num_features = num_init_features
		for i, num_layers in enumerate(block_config):
			block = _DenseBlock(
				num_layers=num_layers,
				num_input_features=num_features,
				bn_size=bn_size,
				growth_rate=growth_rate,
				drop_rate=drop_rate,
				policy = policy[i],
				memory_efficient=memory_efficient
			)
			self.features.add_module('denseblock%d' % (i + 1), block)
			num_features = num_features + num_layers * growth_rate
			if i != len(block_config) - 1:
				trans = _Transition(num_input_features=num_features,
									num_output_features=num_features // 2)
				self.features.add_module('transition%d' % (i + 1), trans)
				num_features = num_features // 2

		# Final batch norm
		self.features.add_module('norm5', nn.BatchNorm2d(num_features))

		# Linear layer
		self.classifier = nn.Linear(num_features, num_classes)

		# Official init from torch repo.
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.constant_(m.bias, 0)

	def forward(self, x):
		if self.inform is None:
			features = self.features(x)
			out = F.relu(features, inplace=True)
			out = F.adaptive_avg_pool2d(out, (1, 1))
			out = torch.flatten(out, 1)
			out = self.classifier(out)
		else:
			flag1=self.inform[0]
			flag2=self.inform[1]
			op = self.features
			out = op.conv0(x)
			out = op.norm0(out)
			out = op.relu0(out)
			out = op.pool0(out)
			for bl_ind in range(len(self.block_config)):
				opp = getattr(op,'denseblock{}'.format(bl_ind+1))
				for Un_ind in range(self.block_config[bl_ind]):
					oppp = getattr(opp,'denselayer{}'.format(Un_ind+1))
					if self.policy[bl_ind][Un_ind] != 0:
						yin = out
						out =  oppp.conv1(oppp.relu1(oppp.norm1(out)))
						out =  oppp.conv2(oppp.relu2(oppp.norm2(out)))
						out_blc = out
						out = torch.cat((yin, out), 1)
					else:
						d0, d2, d3 = out.shape[0], out.shape[2], out.shape[3]
						oppp = torch.nn.Identity()
						remove_part = torch.zeros((d0, 32, d2, d3)) #grouph rates
						remove_part = remove_part.to(device)
						out = torch.cat((out, remove_part), 1)
					if bl_ind==flag1 and Un_ind==flag2 :
						return out
				if bl_ind != len(self.block_config)-1:
					opt = getattr(op,'transition{}'.format(bl_ind+1))
					out = opt(out)
		return out

def densenet121(dataset, policy, model_path, pretrained=True,inform=None):
	if dataset == 'ImageNet':
		kwargs = {"growth_rate": 32, "block_config": (6, 12, 24, 16),
				"num_init_features": 64, "bn_size": 4, "drop_rate": 0, "num_classes": 1000, "memory_efficient": False}
		model = DenseNet(policy = policy,inform =inform, **kwargs)
	else:
		raise ValueError('Check model resnet50 for other dataset')

	if pretrained:
		state_dict = torch.load(model_path)
		pattern = re.compile(
			r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
		for key in list(state_dict.keys()):
			res = pattern.match(key)
			if res:
				new_key = res.group(1) + res.group(2)
				state_dict[new_key] = state_dict[key]
				del state_dict[key]
		model.load_state_dict(state_dict, strict=False)
	return model
