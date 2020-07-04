This is the implement of PED: an algorithm that prune deep nerual networks with Energy Dependence information.
 
-data/
  - CIFAR10/
  - CIFAR100/
  - tiny-imagenet-200/
  - SVHN/
  - ImageNet/
-output/
  - information/
	- dataname_archname_scorename_stage.npy
  - model/
	- dataname_archname_scorename_policymode_stage.pt
	- dataname_archname_scorename_policymode_stage.pth
	- dataname_archname_scorename_policymode_stage_best_pth.tar
	- dataname_archname_scorename_policymode_stage_checkpoint.pth.tar
	- ...
  - policy/
	- dataname_archname_scorename_policymode_stage.npy
	- ...
  - result/
	- prune_result.pkl
	- test_result.pkl
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

  - generate_run.py
  - run.sh
  - show.py

The Example of running one stage, 3 steps
1. Calculate the information metrics to measure the dependence of output on each units
python inform.py --device cuda --pin_memory --dataset CIFAR10 --model resnet --arch resnet56 --score_name energy --policy_mode ckmeans --stage 1

2. Cluster units according to values of the information metric
python prune.py --device cuda --world_size 4 --parallel --pin_memory --dataset CIFAR10 --model resnet --arch resnet56 --score_name energy --policy_mode ckmeans --hyper 15 20 26 --stage 1
xN times

python prune.py --device cuda --world_size 4 --parallel --pin_memory --dataset CIFAR10 --model resnet --arch resnet56 --score_name energy --policy_mode ckmeans --hyper 17 --save_policy --stage 1

3. Retrain the pruned model with warm initialization
python retrain.py --device cuda --world_size 4 --parallel --pin_memory --dataset CIFAR10 --model resnet --arch resnet56 --score_name energy --policy_mode ckmeans --scheduler_name StepLR --step_size 25 --weight_decay 5e-4 --stage 1

Some notations:
(1) mt_score x mt_batch x batchsize_score = number of images random selected for calculating the information metrics. It should cover the whole dataset (n images in train dataset), but considering the time complexity it may cover \sqrt(n) images.
(3) Try --back
(2) retraining tricks, batch_size, StepLR(--weight_decay, --step_size), MultiStepLR(--milestones)
