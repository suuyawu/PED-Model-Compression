#!/bin/bash
CUDA_VISIBLE_DEVICES="0" python Flops_Counting.py --pin_memory --model densenet --dataset CIFAR10 --score_name energy --policy_mode ckmeans --stage 1&
CUDA_VISIBLE_DEVICES="1" python Flops_Counting.py --pin_memory --model resnet --dataset CIFAR10 --score_name energy --policy_mode ckmeans --stage 1&
