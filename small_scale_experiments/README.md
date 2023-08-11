## PyTorch implementation of small-scale experiments presented in our paper: Contrastive Attraction and Contrastive Repulsion for Representation Learning

### 1.Requirements:
pytorch==1.6.0;
tensorboard==2.3.0;
detectron2==0.3;

### 2. Sturcture of code
(1) The codes of CIFAR-10,CIFAR-100,STL-10 experiments in "examples".

(2) The CACR loss of small-scale datasets is in "losses".

### 3. Example Usage

(1) Run the codes on CIFAR-10 (./examples/cifar10/)
Training: 
```
python main_ours.py --Ny 4 --Ns 128 --alpha 1.0 --beta 1.0 --tau_pos 1.0 --tau_neg 0.9 --gpus [gpu_num]
```

Testing: 
```
python linear_eval.py --gpus [gpu_num] --encoder_checkpoint [checkpoint_pth]
```

(2) Run the codes on CIFAR-100 (./examples/cifar100/)
Training: 
```
python main_ours.py --Ny 4 --Ns 128 --alpha 1.0 --beta 1.0 --tau_pos 1.0 --tau_neg 2.0 --gpus [gpu_num]
```
Testing: 
```
python linear_eval.py --gpus [gpu_num] --encoder_checkpoint [checkpoint_pth]
```

(3) Run the codes on STL-10 (./examples/stl10/)
Training: 
```
python main_ours.py --Ny 4 --Ns 128 --alpha 1.0 --beta 1.0 --tau_pos 1.0 --tau_neg 2.0 --gpus [gpu_num]
```

Testing: 
```
python linear_eval.py --gpus [gpu_num] --encoder_checkpoint [checkpoint_pth]
```

### Acknowledgements
This code benefits a lot from previous works like [AlignUniform](https://github.com/SsnL/align_uniform), we would like to thank them here. 



