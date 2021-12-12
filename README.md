# Learning Pruning-Friendly Networks via Frank-Wolfe: One-Shot, Any-Sparsity, and No Retraining 

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Code used for paper: [*Learning Pruning-Friendly Networks via Frank-Wolfe: One-Shot, Any-Sparsity, and No Retraining*](https://openreview.net/pdf?id=O1DEtITim__).

Lu Miao\*, Xiaolong Luo\*, Tianlong Chen, Wuyang Chen, Dong Liu, Zhangyang Wang



## Overview

A novel framework to train a large deep neural network (DNN) for only *once*, which can then be pruned to *any sparsity ratio* to preserve competitive accuracy *without any re-training*. We propose a sparsity-aware one-shot pruning method based on K-sparse polytope constraint and Stochastic Frank-Wolfe (SFW) optimizer. We also present the first *learning-based* initialization scheme specifically for boosting SFW-based DNN training.



## Reproduce

### Preliminary

**Required environment**

- pytorch >= 1.5.0
- torchvision

### Reproducing details

The following codes can reproduce the experiments involved in the paper.

**SFW training for one-shot pruning**

The following code is the training step in SFW-pruning framework.

```python
python -u train_prune.py
					--data cifar10
					--arch ResNet18
					--optimizer SFW
					--constraint k_sparse_constraints
					--lr 1.0
					--lr_scheme dynamic_change
					--momentum 0.9
					--weight_decay 0
					--k_sparseness 10
					--k_frac 0.05
					--tau 15
					--mode initialization
					--rescale gradient
					--sfw_init 0 
					--train_batchsize 128
					--test_batchsize 128
					--epoch_num 180
					--color_channel 3
					--gpu -1
```

The log file is saved in `/saved_logs/SFW_one_shot_prune/`. The trained model is saved in `/saved_models/`.

**Test pruning performance**

The following code conducts (unstructured) pruning and tests the performance of the pruned DNN. Pruning ratios are 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 91%, 92%, 93%, 94%, 95%, 96%, 97%, 98%, 99%.
```python
python -u test_prune.py
					--data cifar10
					--arch ResNet18
					--optimizer SFW
					--constraint k_sparse_constraints
					--lr 1.0
					--lr_scheme dynamic_change
					--momentum 0.9
					--weight_decay 0
					--k_sparseness 10
					--k_frac 0.05
					--tau 15
					--mode initialization
					--rescale gradient
					--sfw_init 0 
					--train_batchsize 128
					--test_batchsize 128
					--epoch_num 180
					--color_channel 3
					--gpu -1
```

The argument choices are parallel with those of `train_prune.py`. The log file is saved in `/saved_logs/SFW_prune_test/`. 

**Check DNN weight distribution**

The following code checks out the weight distribution of the DNN. 

```python
python -u test_weight_distribution.py
					--data cifar10
					--arch ResNet18
					--optimizer SFW
					--constraint k_sparse_constraints
					--lr 1.0
					--lr_scheme dynamic_change
					--momentum 0.9
					--weight_decay 0
					--k_sparseness 10
					--k_frac 0.05
					--tau 15
					--mode initialization
					--rescale gradient
					--sfw_init 0 
					--train_batchsize 128
					--test_batchsize 128
					--epoch_num 180
					--color_channel 3
					--gpu -1
```

The argument choices are also parallel with those of `train_prune.py`. The log file is saved in `/saved_logs/weight_distribution/`. 

**Optional argument choices**

Some optional argument choices are as follows.

```python
optional arguments:
  				--data cifar10 | cifar100 | mnist | svhn | tiny
  				--arch ResNet18 | VGG16 | Mlp
      		--optimizer SFW |SGD
					--constraint k_sparse_constraints | l2_constraints | unconstraints
					--lr 1.0 (float between 0 and 1)
					--lr_scheme dynamic_change | decrease_3 | keep
					--momentum 0.9 (recommand)
					--weight_decay 0 (recommand)
					--k_sparseness 10 (equals to the number of labels)
					--k_frac 0.05 | 0.01 | 0.1
					--tau 15 | 5 | 10 | 20 
					--mode initialization | diameter | radius | None
					--rescale gradient | diameter | None
					--sfw_init 0 | 1
					--train_batchsize 128 
					--test_batchsize 128
					--epoch_num 180
					--color_channel 3
					--gpu -1 (GPU id to use)
```

If use the dataset 'Tiny-Imagenet', please download the dataset to `/data/tiny_imagenet_200/`.



## Citation

```
TBD
```



