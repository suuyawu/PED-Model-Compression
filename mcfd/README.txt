## Model-free Energy Distance for Pruning Deep Neural Networks
We measure new model-free information between the feature maps and the output of the network to compressing Deep Neural Networks (DNNs). Model-freeness of our information measure guarantees that no parametric assumptions on the feature distribution are required.

## Method
In this implementation, we focus on pruning networks with skip-connections by quantifying how informative of the units within the skip-connections about the output of the model, e.g., ReseNet and DenseNet.

## Code structure
-data/
  - CIFAR10/
  - CIFAR100/
  - tiny-imagenet-200/
  - SVHN/
  - ImageNet/
-mcfd/
  - infometrics/
	- Energydist.py
	- FisherInfo.py
	- MutualInfo.py
  - models/
	- resnet.py
	- densenet.py
  - config.py
  - flops.py
  - test_model.py
  - utils.py
  - ckmeans.py
  - inform.py
  - prune.py
  - retrain.py

## Code Examples

## Tests
Describe and show how to run the tests with code examples.

## How to use?
There are three steps for running one pruning stage,
1. Calculate the information metrics to measure the dependence of output on each units
python inform.py --device cuda --pin_memory --dataset CIFAR10 --model resnet --arch resnet56 --score_name energy --policy_mode ckmeans --stage 1

2. Cluster units according to values of the information metric
python prune.py --device cuda --world_size 4 --parallel --pin_memory --dataset CIFAR10 --model resnet --arch resnet56 --score_name energy --policy_mode ckmeans --hyper 15 20 26 --stage 1
xN times

python prune.py --device cuda --world_size 4 --parallel --pin_memory --dataset CIFAR10 --model resnet --arch resnet56 --score_name energy --policy_mode ckmeans --hyper 17 --save_policy --stage 1

3. Retrain the pruned model
python retrain.py --device cuda --world_size 4 --parallel --pin_memory --dataset CIFAR10 --model resnet --arch resnet56 --score_name energy --policy_mode ckmeans --scheduler_name StepLR --step_size 25 --weight_decay 5e-4 --stage 1